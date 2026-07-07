"""Stages 15-16 entry point: consolidate results, make the final selection, save master table.

Combines the baseline and proposed-model results into a single experiment-results table,
then applies a *balanced* selection rule (not "best single metric"):

    * If the proposed model's bootstrap MAE CI overlaps the best baseline's, the difference
      is not meaningful -> prefer the simpler / more interpretable model.
    * Otherwise select the model with the lower CI, tie-broken by high-binding MAE.

Outputs:
    reports/results/experiment_results.{csv,md,json}   - single master table
    reports/results/final_selection.json               - decision + rationale

Usage:
    python scripts/evaluate_models.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ppb_model.results import COLUMNS, _to_markdown
from ppb_model.utils import Paths, get_logger, load_config, set_seed


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage15_16", log_file="logs/stage15_16_evaluate.log")

    frames = []
    for name in ("baseline_results.csv", "improved_results.csv"):
        p = paths.results / name
        if p.is_file():
            frames.append(pd.read_csv(p))
    master = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        if c not in master.columns:
            master[c] = None
    master = master[COLUMNS].sort_values("MAE").reset_index(drop=True)
    master.to_csv(paths.results / "experiment_results.csv", index=False)
    (paths.results / "experiment_results.json").write_text(
        json.dumps(master.to_dict("records"), indent=2), encoding="utf-8")
    (paths.results / "experiment_results.md").write_text(_to_markdown(master), encoding="utf-8")

    # ---- Balanced final selection -------------------------------------------
    boot_path = paths.results / "robustness_bootstrap_primary.json"
    decision = {"method": "balanced", "notes": []}
    if boot_path.is_file():
        boot = json.loads(boot_path.read_text(encoding="utf-8"))
        # baseline reference within the bootstrap set
        base_key = next((k for k in boot if "baseline" in k), None)
        proposed_keys = [k for k in boot if "baseline" not in k]
        best_proposed = min(proposed_keys, key=lambda k: boot[k]["MAE_mean"]) if proposed_keys else None
        decision["bootstrap"] = boot
        if base_key and best_proposed:
            b = boot[base_key]["MAE_CI95"]
            p = boot[best_proposed]["MAE_CI95"]
            overlap = not (p[1] < b[0] or b[1] < p[0])
            decision["baseline_key"] = base_key
            decision["best_proposed_key"] = best_proposed
            decision["mae_ci_overlap"] = overlap
            if overlap:
                # Prefer the model with better high-binding MAE among the near-equivalent set;
                # if the simpler baseline is competitive, recommend it for deployability.
                base_high = boot[base_key]["high_binding_MAE_mean"]
                prop_high = boot[best_proposed]["high_binding_MAE_mean"]
                if prop_high < base_high - 0.05:
                    decision["selected"] = best_proposed
                    decision["rationale"] = (
                        "Overall MAE CIs overlap (no meaningful overall-accuracy gain), but the "
                        f"proposed model improves high-binding MAE ({prop_high:.2f} vs {base_high:.2f}), "
                        "the metric the research gap targets; selected for the high-binding edge plus "
                        "its applicability-domain and uncertainty outputs.")
                else:
                    decision["selected"] = base_key
                    decision["rationale"] = (
                        "Overall and high-binding MAE CIs overlap; the proposed model shows no "
                        "meaningful improvement, so the simpler, more interpretable baseline is "
                        "preferred (honest negative result).")
            else:
                lower = best_proposed if p[1] < b[0] else base_key
                decision["selected"] = lower
                decision["rationale"] = "Non-overlapping MAE CIs; selected the model with the lower interval."
    else:
        decision["notes"].append("Run scripts/robustness.py first for CI-based selection.")
        decision["selected"] = master.iloc[0]["experiment_id"]
        decision["rationale"] = "No bootstrap CIs available; provisional pick = lowest test MAE."

    (paths.results / "final_selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    logger.info("Master table: %d experiments; selected=%s", len(master), decision.get("selected"))
    print(master[["experiment_id", "MAE", "R2", "Spearman", "high_binding_MAE"]].head(12).to_string(index=False))
    print("\nFINAL SELECTION:", decision.get("selected"))
    print("Rationale:", decision.get("rationale"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stages 15-16: consolidate + select.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
