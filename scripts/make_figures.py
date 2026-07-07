"""Stages 10 & 14 entry point: evaluation and applicability-domain figures.

Reads saved artefacts (no model retraining) and renders the reporting figures:
    stage10_pred_vs_obs.png      - predicted vs observed PPB (test)
    stage10_residuals.png        - residuals vs observed (test)
    stage10_mae_by_band.png      - MAE within binding bands (test)
    stage10_model_comparison.png - MAE across baselines + proposed models
    stage14_error_vs_similarity.png - error vs max training similarity (test)
    stage14_ad_boxplot.png       - in- vs out-of-domain error
    stage14_interval_coverage.png - empirical coverage of ensemble-based intervals

Usage:
    python scripts/make_figures.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppb_model.utils import Paths, get_logger, load_config, set_seed


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage10_14", log_file="logs/stage10_14_figures.log")

    pred = pd.read_csv(paths.results / "predictions_final.csv")
    test = pred[pred.data_split == "test"].copy()
    bands = cfg["evaluation"]["binding_bands_percent"]

    # 1. Predicted vs observed (test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test["observed_ppb"], test["predicted_ppb"], s=14, alpha=0.5, color="steelblue")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set_xlabel("Observed PPB (%)"); ax.set_ylabel("Predicted PPB (%)")
    ax.set_title("Predicted vs observed (test, %s)" % test["model_name"].iloc[0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(paths.figures / "stage10_pred_vs_obs.png", dpi=130); plt.close(fig)

    # 2. Residuals vs observed
    fig, ax = plt.subplots(figsize=(7, 5))
    resid = test["predicted_ppb"] - test["observed_ppb"]
    ax.scatter(test["observed_ppb"], resid, s=14, alpha=0.5, color="indianred")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Observed PPB (%)"); ax.set_ylabel("Residual (pred - obs, %)")
    ax.set_title("Residuals vs observed (test)")
    fig.tight_layout(); fig.savefig(paths.figures / "stage10_residuals.png", dpi=130); plt.close(fig)

    # 3. MAE by binding band (test)
    edges = [0.0, *bands, 100.0001]
    labels, maes, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (test["observed_ppb"] >= lo) & (test["observed_ppb"] < hi)
        labels.append(f"[{lo:g},{min(hi,100):g})")
        maes.append(test.loc[m, "absolute_error"].mean() if m.any() else np.nan)
        ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), maes, color="slateblue")
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("PPB band (%)"); ax.set_ylabel("MAE (%)")
    ax.set_title("Test MAE by binding band")
    fig.tight_layout(); fig.savefig(paths.figures / "stage10_mae_by_band.png", dpi=130); plt.close(fig)

    # 4. Model comparison (MAE) across baseline + improved
    frames = []
    for name in ("baseline_results.csv", "improved_results.csv"):
        p = paths.results / name
        if p.is_file():
            frames.append(pd.read_csv(p)[["experiment_id", "MAE", "high_binding_MAE"]])
    comp = pd.concat(frames, ignore_index=True).sort_values("MAE")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(comp["experiment_id"][::-1], comp["MAE"][::-1], color="darkcyan")
    ax.set_xlabel("Test MAE (%)"); ax.set_title("Model comparison (test MAE, lower is better)")
    fig.tight_layout(); fig.savefig(paths.figures / "stage10_model_comparison.png", dpi=130); plt.close(fig)

    # 5. Error vs max training similarity (test)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(test["maximum_training_similarity"], test["absolute_error"], s=14, alpha=0.5, color="teal")
    # binned trend
    bins = np.linspace(test["maximum_training_similarity"].min(), test["maximum_training_similarity"].max(), 8)
    centers = 0.5 * (bins[:-1] + bins[1:])
    binned = [test.loc[(test.maximum_training_similarity >= bins[i]) &
                       (test.maximum_training_similarity < bins[i + 1]), "absolute_error"].mean()
              for i in range(len(bins) - 1)]
    ax.plot(centers, binned, "r-o", lw=2, label="binned mean")
    ax.set_xlabel("Max Tanimoto similarity to training set"); ax.set_ylabel("Absolute error (%)")
    ax.set_title("Error vs training-set similarity (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(paths.figures / "stage14_error_vs_similarity.png", dpi=130); plt.close(fig)

    # 6. AD boxplot in vs out
    fig, ax = plt.subplots(figsize=(6, 5))
    grp = [test.loc[test.in_applicability_domain, "absolute_error"],
           test.loc[~test.in_applicability_domain, "absolute_error"]]
    ax.boxplot(grp, tick_labels=["in-domain", "out-of-domain"], showmeans=True)
    ax.set_ylabel("Absolute error (%)"); ax.set_title("Error by applicability domain (test)")
    fig.tight_layout(); fig.savefig(paths.figures / "stage14_ad_boxplot.png", dpi=130); plt.close(fig)

    # 7. Prediction-interval calibration: raw ensemble std vs split-conformal.
    from scipy import stats as sstats
    valid = pred[pred.data_split == "valid"].copy()

    # (a) ensemble-std interval (uncalibrated) across z.
    z_values = np.linspace(0.5, 3.0, 11)
    nominal = [2 * sstats.norm.cdf(z) - 1 for z in z_values]
    ens_cov = []
    for z in z_values:
        lo = test["predicted_ppb"] - z * test["uncertainty_score"]
        hi = test["predicted_ppb"] + z * test["uncertainty_score"]
        ens_cov.append(float(((test["observed_ppb"] >= lo) & (test["observed_ppb"] <= hi)).mean()))

    # (b) split-conformal interval: half-width = (1-alpha) quantile of |valid residual|.
    valid_abs_resid = (valid["predicted_ppb"] - valid["observed_ppb"]).abs().to_numpy()
    conf_nominal = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    conf_cov, conf_width = [], []
    for lvl in conf_nominal:
        q = float(np.quantile(valid_abs_resid, lvl))
        lo = test["predicted_ppb"] - q
        hi = test["predicted_ppb"] + q
        conf_cov.append(float(((test["observed_ppb"] >= lo) & (test["observed_ppb"] <= hi)).mean()))
        conf_width.append(q)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(nominal, ens_cov, "b-o", label="ensemble std (uncalibrated)")
    ax.plot(conf_nominal, conf_cov, "g-s", label="split-conformal (calibrated)")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (test)")
    ax.set_title("Prediction-interval calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(paths.figures / "stage14_interval_coverage.png", dpi=130); plt.close(fig)

    # summaries at ~95%/90%.
    z95 = 1.96
    lo = test["predicted_ppb"] - z95 * test["uncertainty_score"]
    hi = test["predicted_ppb"] + z95 * test["uncertainty_score"]
    ens_cov95 = float(((test["observed_ppb"] >= lo) & (test["observed_ppb"] <= hi)).mean())
    q90 = float(np.quantile(valid_abs_resid, 0.90))
    conf_cov90 = float(((test["observed_ppb"] >= test["predicted_ppb"] - q90) &
                        (test["observed_ppb"] <= test["predicted_ppb"] + q90)).mean())
    (paths.results / "stage14_coverage.json").write_text(
        json.dumps({"ensemble_coverage_at_1.96sigma": round(ens_cov95, 4),
                    "conformal_target_0.90_halfwidth_pct": round(q90, 3),
                    "conformal_empirical_coverage_at_0.90": round(conf_cov90, 4)}, indent=2),
        encoding="utf-8")

    logger.info("Figures written to %s", paths.figures)
    print("Wrote 7 figures to reports/figures/")
    print(f"Ensemble +/-1.96sigma coverage (test): {ens_cov95:.2%} (uncalibrated)")
    print(f"Split-conformal 90% interval: half-width={q90:.1f}pct, test coverage={conf_cov90:.2%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stages 10 & 14: figures.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
