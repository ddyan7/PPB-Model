"""Stage 11 entry point: robustness via repeated scaffold splits + bootstrap CIs.

Two complementary uncertainty estimates:
    1. Repeated scaffold splits (different seeds) -> mean +/- 95% CI of test metrics,
       showing how much performance depends on which scaffolds land in the test fold.
    2. Bootstrap resampling of the primary-split test set -> 95% CI on each model's MAE,
       to judge whether the proposed model's edge over the baseline is real or noise.

Fixed (already-selected) hyperparameters are reused across repeats; nothing is re-tuned,
and the test folds here are used only for reporting, never for selection.

Outputs:
    reports/results/robustness_repeated_splits.csv
    reports/results/robustness_bootstrap_primary.json
    reports/figures/stage11_model_comparison_ci.png

Usage:
    python scripts/robustness.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppb_model.baselines import make_model
from ppb_model.evaluation import regression_metrics
from ppb_model.features import DescriptorCleaner
from ppb_model.splitting import scaffold_split
from ppb_model.targets import TargetTransformer
from ppb_model.train import FeatureCache
from ppb_model.utils import Paths, get_logger, load_config, set_seed

REPEAT_SEEDS = [42, 43, 44, 45, 46]


def _assemble(cache, tr, va, te, var_thr, corr_thr):
    """Return descriptor and hybrid matrices for a given index split (train-only clean)."""
    cleaner = DescriptorCleaner(var_thr, corr_thr).fit(cache.desc_raw[tr], cache.desc_names)
    d = {s: cleaner.transform(cache.desc_raw[i]) for s, i in (("train", tr), ("valid", va), ("test", te))}
    m = {"train": cache.morgan[tr], "valid": cache.morgan[va], "test": cache.morgan[te]}
    h = {s: np.hstack([d[s], m[s]]) for s in ("train", "valid", "test")}
    return d, h


def _fit_pred(model_name, params, X_tr, y_tr_t, X_te, tf, seed):
    model = make_model(model_name, seed=seed, **params)
    model.fit(X_tr, y_tr_t)
    return tf.inverse(model.predict(X_te)), model


def _bootstrap_mae_ci(y_true, y_pred, n_boot, seed, high_threshold):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes, high_maes = [], []
    hi = y_true >= high_threshold
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ae = np.abs(y_true[idx] - y_pred[idx])
        maes.append(ae.mean())
        hi_b = hi[idx]
        high_maes.append(np.abs(y_true[idx][hi_b] - y_pred[idx][hi_b]).mean() if hi_b.any() else np.nan)
    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_CI95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "high_binding_MAE_mean": float(np.nanmean(high_maes)),
        "high_binding_MAE_CI95": [float(np.nanpercentile(high_maes, 2.5)),
                                  float(np.nanpercentile(high_maes, 97.5))],
    }


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed0 = cfg["project"]["seed"]
    set_seed(seed0)
    paths = Paths.create()
    logger = get_logger("stage11", log_file="logs/stage11_robustness.log")

    df = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    smiles = df["canonical_smiles"].tolist()
    cache = FeatureCache.load(paths.interim / "features.npz")
    y_pct = cache.y_percent

    tuned = json.loads((paths.results / "tuned_hyperparameters.json").read_text(encoding="utf-8"))
    tf = TargetTransformer(cfg["target"]["transform"], float(cfg["target"]["epsilon"]),
                           tuple(cfg["target"]["clip_percent"]))
    var_thr = cfg["features"]["descriptor_variance_threshold"]
    corr_thr = cfg["features"]["descriptor_corr_threshold"]
    high_thr = cfg["evaluation"]["high_binding_threshold_percent"]
    bands = tuple(cfg["evaluation"]["binding_bands_percent"])
    tf_, vf_, tsf_ = cfg["split"]["train_frac"], cfg["split"]["valid_frac"], cfg["split"]["test_frac"]

    # Model specs assessed for robustness.
    specs = {
        "elasticnet_desc(baseline)": ("elasticnet", "desc", {"alpha": 0.01, "l1_ratio": 0.5}),
        "rf_desc(tuned)": ("rf", "desc", tuned["rf_desc"]["best_params"]),
        "xgb_hybrid(tuned)": ("xgb", "hybrid", tuned["xgb_hybrid"]["best_params"]),
        "consensus": ("consensus", "mixed", None),
    }
    consensus_members = [("xgb", "hybrid", tuned["xgb_hybrid"]["best_params"]),
                         ("rf", "desc", tuned["rf_desc"]["best_params"]),
                         ("hgb", "desc", tuned["hgb_desc"]["best_params"])]

    # ---- 1. Repeated scaffold splits ----------------------------------------
    repeat_rows = []
    primary_preds: dict[str, np.ndarray] = {}
    primary_y_test: np.ndarray | None = None
    for rep_i, seed in enumerate(REPEAT_SEEDS):
        sp = scaffold_split(smiles, tf_, vf_, tsf_, seed=seed)
        tr, va, te = sp["train"], sp["valid"], sp["test"]
        y_tr_t = tf.forward(y_pct[tr])
        y_te = y_pct[te]
        d, h = _assemble(cache, tr, va, te, var_thr, corr_thr)
        mats = {"desc": d, "hybrid": h}

        for label, (mname, rep, params) in specs.items():
            if mname == "consensus":
                stack = []
                for cm, crep, cpar in consensus_members:
                    pred, _ = _fit_pred(cm, cpar, mats[crep]["train"], y_tr_t, mats[crep]["test"], tf, seed)
                    stack.append(pred)
                pred = np.mean(stack, axis=0)
            else:
                pred, _ = _fit_pred(mname, params, mats[rep]["train"], y_tr_t, mats[rep]["test"], tf, seed)
            m = regression_metrics(y_te, pred, high_binding_threshold=high_thr, bands=bands)
            repeat_rows.append({"seed": seed, "model": label, "MAE": m["MAE"], "RMSE": m["RMSE"],
                                "R2": m["R2"], "Spearman": m["Spearman"],
                                "high_binding_MAE": m["high_binding_MAE"], "fu_MAE": m["fu_MAE"]})
            if seed == seed0:
                primary_preds[label] = pred
                primary_y_test = y_te
        logger.info("Repeat %d/%d (seed=%d) done", rep_i + 1, len(REPEAT_SEEDS), seed)

    rep_df = pd.DataFrame(repeat_rows)
    agg = rep_df.groupby("model").agg(["mean", "std"]).round(4)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(paths.results / "robustness_repeated_splits.csv", index=False)

    # ---- 2. Bootstrap CIs on the primary split ------------------------------
    n_boot = cfg["evaluation"]["bootstrap_n"]
    boot = {label: _bootstrap_mae_ci(primary_y_test, pred, n_boot, seed0, high_thr)
            for label, pred in primary_preds.items()}
    (paths.results / "robustness_bootstrap_primary.json").write_text(
        json.dumps(boot, indent=2), encoding="utf-8")

    # ---- Figure: repeated-split MAE with spread -----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    order = list(specs.keys())
    means = [rep_df[rep_df.model == m]["MAE"].mean() for m in order]
    stds = [rep_df[rep_df.model == m]["MAE"].std() for m in order]
    ax1.bar(range(len(order)), means, yerr=stds, capsize=5, color="steelblue")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Test MAE (%)")
    ax1.set_title(f"Repeated scaffold splits (n={len(REPEAT_SEEDS)})\nmean +/- SD")

    hmeans = [rep_df[rep_df.model == m]["high_binding_MAE"].mean() for m in order]
    hstds = [rep_df[rep_df.model == m]["high_binding_MAE"].std() for m in order]
    ax2.bar(range(len(order)), hmeans, yerr=hstds, capsize=5, color="indianred")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("High-binding (>=90%) MAE")
    ax2.set_title("High-binding MAE across repeats")
    fig.tight_layout()
    fig.savefig(paths.figures / "stage11_model_comparison_ci.png", dpi=130)
    plt.close(fig)

    print("=== Repeated scaffold splits (mean +/- SD over %d seeds) ===" % len(REPEAT_SEEDS))
    for m in order:
        sub = rep_df[rep_df.model == m]
        print(f"{m:28s} MAE={sub['MAE'].mean():.3f}+/-{sub['MAE'].std():.3f} | "
              f"highMAE={sub['high_binding_MAE'].mean():.3f}+/-{sub['high_binding_MAE'].std():.3f} | "
              f"Spearman={sub['Spearman'].mean():.3f}")
    print("\n=== Bootstrap 95% CI on primary-split test MAE ===")
    for label, b in boot.items():
        print(f"{label:28s} MAE={b['MAE_mean']:.3f} CI[{b['MAE_CI95'][0]:.3f},{b['MAE_CI95'][1]:.3f}] | "
              f"highMAE={b['high_binding_MAE_mean']:.3f} "
              f"CI[{b['high_binding_MAE_CI95'][0]:.3f},{b['high_binding_MAE_CI95'][1]:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 11: robustness (repeated splits + bootstrap).")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
