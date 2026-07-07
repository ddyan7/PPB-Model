"""Stage 12 entry point: controlled ablation studies.

Each ablation changes exactly one factor, holding the model and split fixed, so the
effect is attributable. All comparisons use the same evaluation harness and metrics.

    A. Representation : xgb (default) on {descriptors, morgan, maccs, hybrid}, logit, scaffold
    B. Target transform : xgb (default) on descriptors x {none, fraction_unbound, log_fu, logit, lnKa}
    C. Split : rf (tuned) on descriptors+logit, {scaffold, random}
    D. Single vs consensus : from improved_results (reference)
    E. Applicability domain : test MAE all vs in-domain only (from predictions_final)

Outputs:
    reports/results/ablation_results.csv / .md
    reports/results/ablation_summary.json

Usage:
    python scripts/ablation.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ppb_model.data import load_clean_and_split
from ppb_model.results import _to_markdown
from ppb_model.train import FeatureCache, run_experiment
from ppb_model.utils import Paths, get_logger, load_config, set_seed


def _row(ablation, factor, res):
    m = res.test_metrics
    return {"ablation": ablation, "factor": factor, "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4), "R2": round(m["R2"], 4),
            "Spearman": round(m["Spearman"], 4), "high_binding_MAE": round(m["high_binding_MAE"], 4),
            "fu_MAE": round(m["fu_MAE"], 5)}


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage12", log_file="logs/stage12_ablation.log")

    cache = FeatureCache.load(paths.interim / "features.npz")
    scaffold = load_clean_and_split(cfg, "scaffold")
    random_ = load_clean_and_split(cfg, "random")
    tuned = json.loads((paths.results / "tuned_hyperparameters.json").read_text(encoding="utf-8"))

    rows = []

    # A. Representation (xgb default, logit, scaffold)
    for rep in ("descriptors", "morgan", "maccs", "hybrid"):
        res = run_experiment(model_name="xgb", representation=rep, transform_method="logit",
                             cache=cache, idx=scaffold["idx"], config=cfg)
        rows.append(_row("A_representation", rep, res))

    # B. Target transform (xgb default, descriptors, scaffold)
    for tform in ("none", "fraction_unbound", "log_fu", "logit", "lnKa"):
        res = run_experiment(model_name="xgb", representation="descriptors", transform_method=tform,
                             cache=cache, idx=scaffold["idx"], config=cfg)
        rows.append(_row("B_transform", tform, res))

    # C. Split method (rf tuned, descriptors, logit)
    rf_params = tuned["rf_desc"]["best_params"]
    for split_name, data in (("scaffold", scaffold), ("random", random_)):
        res = run_experiment(model_name="rf", representation="descriptors", transform_method="logit",
                             cache=cache, idx=data["idx"], config=cfg, model_params=rf_params)
        rows.append(_row("C_split", split_name, res))

    # D. Single vs consensus (reference from improved_results)
    imp = pd.read_csv(paths.results / "improved_results.csv")
    for _, r in imp.iterrows():
        rows.append({"ablation": "D_single_vs_consensus", "factor": r["experiment_id"],
                     "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"], "Spearman": r["Spearman"],
                     "high_binding_MAE": r["high_binding_MAE"], "fu_MAE": r["fraction_unbound_MAE"]})

    # E. Applicability domain (from final predictions, test split)
    pred = pd.read_csv(paths.results / "predictions_final.csv")
    test = pred[pred.data_split == "test"]
    ad_summary = {
        "test_MAE_all": round(test["absolute_error"].mean(), 4),
        "test_MAE_in_domain": round(test[test.in_applicability_domain]["absolute_error"].mean(), 4),
        "test_MAE_out_domain": round(test[~test.in_applicability_domain]["absolute_error"].mean(), 4),
        "in_domain_fraction": round(test["in_applicability_domain"].mean(), 4),
        "uncertainty_error_corr": round(test["uncertainty_score"].corr(test["absolute_error"]), 4),
    }
    rows.append({"ablation": "E_applicability_domain", "factor": "test_all",
                 "MAE": ad_summary["test_MAE_all"], "RMSE": None, "R2": None, "Spearman": None,
                 "high_binding_MAE": None, "fu_MAE": None})
    rows.append({"ablation": "E_applicability_domain", "factor": "test_in_domain_only",
                 "MAE": ad_summary["test_MAE_in_domain"], "RMSE": None, "R2": None, "Spearman": None,
                 "high_binding_MAE": None, "fu_MAE": None})

    df = pd.DataFrame(rows)
    df.to_csv(paths.results / "ablation_results.csv", index=False)
    (paths.results / "ablation_results.md").write_text(_to_markdown(df), encoding="utf-8")
    (paths.results / "ablation_summary.json").write_text(
        json.dumps({"applicability_domain": ad_summary}, indent=2), encoding="utf-8")

    logger.info("Ablation complete: %d rows", len(df))
    for ab in df["ablation"].unique():
        print(f"\n--- {ab} ---")
        print(df[df.ablation == ab].drop(columns="ablation").to_string(index=False))
    print("\nAD summary:", json.dumps(ad_summary))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 12: ablation studies.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
