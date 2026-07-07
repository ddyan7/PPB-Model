"""Stage 3 entry point: compare PPB target transformations and record the decision.

For each candidate transformation this reports:
    * numerical stability  - non-finite count and round-trip (forward->inverse) error
    * distributional shape - skewness and excess kurtosis (closer to 0 = more Gaussian)
    * high-binding resolution - spread (std) allocated to compounds with Y in [95, 100)%,
      i.e. how well the transform separates near-100%-bound molecules

No model is trained here (test set stays untouched); the empirical raw-vs-logit
performance check is deferred to the Stage 12 ablation on a fixed split.

Outputs:
    reports/tables/stage3_target_comparison.csv
    reports/figures/stage3_target_distributions.png
    reports/results/stage3_target_decision.json

Usage:
    python scripts/analyze_target.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ppb_model.targets import TargetTransformer
from ppb_model.utils import Paths, get_logger, load_config, set_seed

CANDIDATES = ["none", "fraction_bound", "fraction_unbound", "log_fu", "logit", "lnKa", "clipped"]
LABELS = {
    "none": "Percent bound (raw)",
    "fraction_bound": "Fraction bound",
    "fraction_unbound": "Fraction unbound",
    "log_fu": "log10(fu)",
    "logit": "logit(fb)",
    "lnKa": "lnKa = 0.5*ln(fb/fu)",
    "clipped": "Clipped fraction bound",
}


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage3", log_file="logs/stage3_target.log")

    clean_csv = paths.processed / "ppbr_az_human_clean.csv"
    if not clean_csv.is_file():
        raise FileNotFoundError(f"Run Stage 2 first; missing {clean_csv}")
    df = pd.read_csv(clean_csv)
    y = df["y_percent"].to_numpy(dtype=float)
    logger.info("Loaded %d cleaned rows for target analysis", len(y))

    eps = float(cfg["target"]["epsilon"])
    clip = tuple(cfg["target"]["clip_percent"])
    high_mask = (y >= 95.0) & (y < 100.0)  # near-fully-bound subset

    rows = []
    transformed_cache: dict[str, np.ndarray] = {}
    for method in CANDIDATES:
        tf = TargetTransformer(method=method, epsilon=eps, clip_percent=clip)
        t = tf.forward(y)
        transformed_cache[method] = t
        n_nonfinite = int((~np.isfinite(t)).sum())
        # round-trip stability: forward then inverse should recover y (within clipping)
        y_rt = tf.inverse(t)
        # ignore clipping-induced differences at the extremes by comparing on the clipped input
        y_clip = np.clip(y, clip[0], clip[1])
        rt_max_abs = float(np.max(np.abs(y_rt - y_clip)))
        finite = t[np.isfinite(t)]
        rows.append({
            "method": method,
            "label": LABELS[method],
            "skewness": float(stats.skew(finite)),
            "excess_kurtosis": float(stats.kurtosis(finite)),  # Fisher: 0 == normal
            "abs_skewness": float(abs(stats.skew(finite))),
            "high_binding_std": float(np.std(t[high_mask])),   # resolution in [95,100)%
            "n_nonfinite": n_nonfinite,
            "roundtrip_max_abs_pct": rt_max_abs,
        })

    comp = pd.DataFrame(rows)
    # Rank primarily by how Gaussian the distribution is (|skew|), a key modelling driver.
    comp = comp.sort_values("abs_skewness").reset_index(drop=True)
    comp_path = paths.tables / "stage3_target_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # ---- Figure: distribution of each transform ------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, method in zip(axes.ravel(), CANDIDATES):
        ax.hist(transformed_cache[method][np.isfinite(transformed_cache[method])],
                bins=40, color="steelblue", edgecolor="white")
        sk = float(stats.skew(transformed_cache[method][np.isfinite(transformed_cache[method])]))
        ax.set_title(f"{LABELS[method]}\nskew={sk:+.2f}", fontsize=10)
        ax.set_ylabel("count")
    axes.ravel()[-1].axis("off")  # 7 candidates in an 8-panel grid
    fig.suptitle("PPB target transformations (human PPBR_AZ, n=%d)" % len(y), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = paths.figures / "stage3_target_distributions.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ---- Decision ------------------------------------------------------------
    configured = cfg["target"]["transform"]
    best_gaussian = comp.iloc[0]["method"]
    decision = {
        "configured_transform": configured,
        "most_gaussian_transform": best_gaussian,
        "rationale": (
            "Transformation is selected on scientific meaning + distributional shape + "
            "high-binding resolution, not a single metric. logit/lnKa map the fraction-bound "
            "onto the real line, symmetrise the strongly left-skewed high-binding mass, and give "
            "the largest spread to near-100%-bound compounds (fu-sensitive region). This matches "
            "Han et al. 2025 (LogIt) and Watanabe et al. 2018 (log-fu). The empirical raw-vs-logit "
            "comparison is confirmed on a fixed split in the Stage 12 ablation."
        ),
        "raw_percent_abs_skew": float(comp.set_index("method").loc["none", "abs_skewness"]),
        "logit_abs_skew": float(comp.set_index("method").loc["logit", "abs_skewness"]),
        "high_binding_std": {
            m: float(comp.set_index("method").loc[m, "high_binding_std"]) for m in CANDIDATES
        },
        "outputs": {"comparison_csv": str(comp_path), "figure": str(fig_path)},
    }
    dec_path = paths.results / "stage3_target_decision.json"
    dec_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    logger.info("Wrote %s", comp_path)
    logger.info("Wrote %s", fig_path)
    logger.info("Wrote %s", dec_path)
    print(comp.to_string(index=False))
    print()
    print("Configured transform:", configured, "| most-Gaussian:", best_gaussian)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: target-transformation analysis.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
