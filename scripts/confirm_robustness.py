"""Confirm the augmentation gain is real: repeated scaffold splits + paired bootstrap.

Compares the ORIGINAL consensus (PPBR_AZ-only training) against the AUGMENTED consensus
(PPBR_AZ + scaffold-safe Ingle) using their already-selected hyperparameters (no re-tuning):

    1. Repeated scaffold splits (5 seeds): both models refit per split and scored on that
       split's held-out PPBR_AZ test -> mean +/- SD and per-split paired win/loss.
    2. Paired bootstrap on the primary split: resample the shared test compounds and take
       MAE(augmented) - MAE(original); a 95% CI below 0 means a real improvement.

Non-overlapping Ingle features are computed once; per split, only compounds whose scaffold
is absent from that split's valid/test are added to training (leakage-safe).

Outputs:
    reports/results/confirm_robustness.json

Usage:
    python scripts/confirm_robustness.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ppb_model.baselines import make_model
from ppb_model.evaluation import regression_metrics
from ppb_model.features import DescriptorCleaner, compute_descriptor_matrix, morgan_matrix
from ppb_model.splitting import bemis_murcko_scaffold, scaffold_split
from ppb_model.standardisation import standardise_smiles
from ppb_model.targets import TargetTransformer
from ppb_model.train import FeatureCache
from ppb_model.utils import Paths, get_logger, load_config, resolve_path, set_seed

REPEAT_SEEDS = [42, 43, 44, 45, 46]
MEMBERS = [("xgb_hybrid", "xgb", "hybrid"), ("rf_desc", "rf", "descriptors"),
           ("hgb_desc", "hgb", "descriptors")]


def _load_ingle_nonoverlap(config: dict[str, Any], paths):
    """Standardise Ingle, drop PPBR_AZ InChIKey overlaps; return frame + features (computed once)."""
    dcfg = config["data"]
    ingle_data_path = resolve_path(dcfg["ingle_data_csv"])
    if not ingle_data_path.is_file():
        raise FileNotFoundError(f"Ingle dataset not found: {ingle_data_path}")
    raw = pd.read_csv(ingle_data_path)
    ppbr_keys = set(pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")["inchikey"].dropna())
    recs = []
    for r in raw.itertuples(index=False):
        fub = pd.to_numeric(getattr(r, "Fub"), errors="coerce")
        if pd.isna(fub) or not (0 <= fub <= 1):
            continue
        res = standardise_smiles(getattr(r, "SMILES"))
        if not res.ok or res.inchikey in ppbr_keys:
            continue
        recs.append({"canonical_smiles": res.canonical_smiles, "inchikey": res.inchikey,
                     "scaffold": bemis_murcko_scaffold(res.canonical_smiles) or "__acyclic__",
                     "ppb": 100.0 * (1.0 - float(fub))})
    df = pd.DataFrame(recs).drop_duplicates("inchikey").reset_index(drop=True)
    return df


def _consensus_predict(desc_raw_tr, morgan_tr, y_tr_pct, params, cleaner, tf,
                       desc_raw_te, morgan_te):
    """Fit the 3-member consensus and return percent predictions on the test features."""
    y_tr_t = tf.forward(y_tr_pct)
    logit_preds = []
    for key, mname, rep in MEMBERS:
        X_tr = cleaner.transform(desc_raw_tr)
        X_te = cleaner.transform(desc_raw_te)
        if rep == "hybrid":
            X_tr = np.hstack([X_tr, morgan_tr])
            X_te = np.hstack([X_te, morgan_te])
        model = make_model(mname, seed=42, **params[key])
        model.fit(X_tr, y_tr_t)
        logit_preds.append(model.predict(X_te))
    return tf.inverse(np.vstack(logit_preds).mean(axis=0))


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("confirm", log_file="logs/confirm_robustness.log")

    cache = FeatureCache.load(paths.interim / "features.npz")
    ppbr = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    smiles = ppbr["canonical_smiles"].tolist()
    y_all = cache.y_percent
    fcfg, tcfg = cfg["features"], cfg["target"]
    tf = TargetTransformer(tcfg["transform"], float(tcfg["epsilon"]), tuple(tcfg["clip_percent"]))
    mk = dict(high_binding_threshold=cfg["evaluation"]["high_binding_threshold_percent"],
              bands=tuple(cfg["evaluation"]["binding_bands_percent"]))
    var_thr, corr_thr = fcfg["descriptor_variance_threshold"], fcfg["descriptor_corr_threshold"]
    m_r, m_b = fcfg["morgan"]["radius"], fcfg["morgan"]["n_bits"]

    # Hyperparameters (already selected; not re-tuned here).
    orig_params = {k: v["best_params"] for k, v in
                   json.loads((paths.results / "tuned_hyperparameters.json").read_text()).items()
                   if k in ("xgb_hybrid", "rf_desc", "hgb_desc")}
    aug = json.loads((paths.results / "augmented_results.json").read_text())
    aug_params = {k: v["best_params"] for k, v in aug["tuned"].items()}

    # Non-overlapping Ingle, features computed once.
    ingle = _load_ingle_nonoverlap(cfg, paths)
    logger.info("Non-overlapping Ingle compounds: %d", len(ingle))
    ingle_desc, _ = compute_descriptor_matrix(ingle["canonical_smiles"].tolist())
    ingle_morgan = morgan_matrix(ingle["canonical_smiles"].tolist(), m_r, m_b).astype(cache.morgan.dtype)
    ingle_scaf = ingle["scaffold"].to_numpy()
    ingle_y = ingle["ppb"].to_numpy()

    rows = []
    primary_preds = {}
    for seed in REPEAT_SEEDS:
        sp = scaffold_split(smiles, cfg["split"]["train_frac"], cfg["split"]["valid_frac"],
                            cfg["split"]["test_frac"], seed=seed)
        tr, te = sp["train"], sp["test"]
        te_scaf = {bemis_murcko_scaffold(smiles[i]) or "__acyclic__" for i in np.concatenate([sp["valid"], te])}
        safe = np.array([s not in te_scaf for s in ingle_scaf])
        y_te = y_all[te]

        # Original: train on PPBR_AZ train only.
        cln_o = DescriptorCleaner(var_thr, corr_thr).fit(cache.desc_raw[tr], cache.desc_names)
        pred_o = _consensus_predict(cache.desc_raw[tr], cache.morgan[tr], y_all[tr], orig_params,
                                    cln_o, tf, cache.desc_raw[te], cache.morgan[te])
        # Augmented: train on PPBR_AZ train + scaffold-safe Ingle.
        desc_tr_a = np.vstack([cache.desc_raw[tr], ingle_desc[safe]])
        morgan_tr_a = np.vstack([cache.morgan[tr], ingle_morgan[safe]])
        y_tr_a = np.concatenate([y_all[tr], ingle_y[safe]])
        cln_a = DescriptorCleaner(var_thr, corr_thr).fit(desc_tr_a, cache.desc_names)
        pred_a = _consensus_predict(desc_tr_a, morgan_tr_a, y_tr_a, aug_params,
                                    cln_a, tf, cache.desc_raw[te], cache.morgan[te])

        mo = regression_metrics(y_te, pred_o, **mk)
        ma = regression_metrics(y_te, pred_a, **mk)
        rows.append({"seed": seed, "n_ingle_added": int(safe.sum()),
                     "orig_MAE": mo["MAE"], "aug_MAE": ma["MAE"],
                     "orig_R2": mo["R2"], "aug_R2": ma["R2"],
                     "orig_Spearman": mo["Spearman"], "aug_Spearman": ma["Spearman"],
                     "orig_highMAE": mo["high_binding_MAE"], "aug_highMAE": ma["high_binding_MAE"]})
        logger.info("seed %d: orig MAE=%.3f aug MAE=%.3f (+%d Ingle)", seed, mo["MAE"], ma["MAE"], int(safe.sum()))
        if seed == cfg["project"]["seed"]:
            primary_preds = {"y_te": y_te, "orig": pred_o, "aug": pred_a}

    rep = pd.DataFrame(rows)

    def agg(col):
        return {"mean": round(float(rep[col].mean()), 4), "sd": round(float(rep[col].std()), 4)}

    repeated = {m: {"original": agg(f"orig_{m}"), "augmented": agg(f"aug_{m}")}
                for m in ("MAE", "R2", "Spearman", "highMAE")}
    mae_deltas = (rep["aug_MAE"] - rep["orig_MAE"])
    repeated["MAE_paired_delta_per_split"] = [round(float(d), 4) for d in mae_deltas]
    repeated["MAE_augmented_wins"] = int((mae_deltas < 0).sum())

    # Paired bootstrap on the primary split.
    y_te = primary_preds["y_te"]
    ae_o = np.abs(y_te - primary_preds["orig"])
    ae_a = np.abs(y_te - primary_preds["aug"])
    hi = y_te >= mk["high_binding_threshold"]
    rng = np.random.default_rng(cfg["project"]["seed"])
    n = len(y_te)
    dmae, dhigh = [], []
    for _ in range(cfg["evaluation"]["bootstrap_n"]):
        idx = rng.integers(0, n, n)
        dmae.append(ae_a[idx].mean() - ae_o[idx].mean())
        hb = hi[idx]
        if hb.any():
            dhigh.append(ae_a[idx][hb].mean() - ae_o[idx][hb].mean())
    boot = {
        "delta_MAE_mean": round(float(np.mean(dmae)), 4),
        "delta_MAE_CI95": [round(float(np.percentile(dmae, 2.5)), 4), round(float(np.percentile(dmae, 97.5)), 4)],
        "delta_MAE_prob_improvement": round(float(np.mean(np.array(dmae) < 0)), 4),
        "delta_highMAE_mean": round(float(np.mean(dhigh)), 4),
        "delta_highMAE_CI95": [round(float(np.percentile(dhigh, 2.5)), 4), round(float(np.percentile(dhigh, 97.5)), 4)],
    }

    summary = {"repeated_splits": repeated, "paired_bootstrap_primary": boot,
               "n_seeds": len(REPEAT_SEEDS)}
    (paths.results / "confirm_robustness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Repeated scaffold splits (mean +/- SD over %d seeds) ===" % len(REPEAT_SEEDS))
    for m in ("MAE", "R2", "Spearman", "highMAE"):
        o, a = repeated[m]["original"], repeated[m]["augmented"]
        print(f"{m:10s} original {o['mean']:.3f}+/-{o['sd']:.3f} | augmented {a['mean']:.3f}+/-{a['sd']:.3f}")
    print(f"\nPer-split MAE delta (aug-orig): {repeated['MAE_paired_delta_per_split']}")
    print(f"Augmented wins MAE on {repeated['MAE_augmented_wins']}/{len(REPEAT_SEEDS)} splits")
    print("\n=== Paired bootstrap (primary split) ===")
    print(f"delta MAE = {boot['delta_MAE_mean']:+.3f}  95% CI [{boot['delta_MAE_CI95'][0]:+.3f}, "
          f"{boot['delta_MAE_CI95'][1]:+.3f}]  P(improvement)={boot['delta_MAE_prob_improvement']:.2f}")
    print(f"delta high-binding MAE = {boot['delta_highMAE_mean']:+.3f}  95% CI "
          f"[{boot['delta_highMAE_CI95'][0]:+.3f}, {boot['delta_highMAE_CI95'][1]:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirm augmentation robustness.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
