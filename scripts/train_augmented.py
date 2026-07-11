"""Improvement experiment: data augmentation (Ingle) + scaffold-grouped-CV tuning.

Tests the top recommendation honestly, with a leakage-safe design:
    * Test set = the UNCHANGED PPBR_AZ scaffold test (243) -> direct before/after comparison.
    * Training is augmented with non-overlapping Ingle compounds, EXCLUDING any whose
      Bemis-Murcko scaffold appears in the PPBR_AZ valid/test (so augmentation cannot leak).
    * Hyperparameters tuned with scaffold-GROUPED 3-fold CV (a stable selection signal),
      replacing the original single noisy validation fold.

Reports whether test / high-binding MAE improve versus the original model - including if
they do not (mixing a second assay source can add noise).

Outputs:
    reports/results/augmented_results.json

Usage:
    python scripts/train_augmented.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ppb_model.baselines import make_model
from ppb_model.evaluation import regression_metrics
from ppb_model.features import DescriptorCleaner, compute_descriptor_matrix, morgan_matrix
from ppb_model.splitting import bemis_murcko_scaffold
from ppb_model.standardisation import standardise_smiles
from ppb_model.targets import TargetTransformer
from ppb_model.train import FeatureCache
from ppb_model.tuning import tune_model_grouped
from ppb_model.utils import Paths, get_logger, load_config, resolve_path, set_seed

MEMBERS = [("xgb_hybrid", "xgb", "hybrid"), ("rf_desc", "rf", "descriptors"),
           ("hgb_desc", "hgb", "descriptors")]


def _load_ingle_safe(cfg, paths, ppbr_keys, forbidden_scaffolds, logger):
    """Return standardised Ingle compounds that are non-overlapping and scaffold-safe."""
    dcfg = cfg["data"]
    ingle_data_path = resolve_path(dcfg["ingle_data_csv"])
    if not ingle_data_path.is_file():
        raise FileNotFoundError(f"Ingle dataset not found: {ingle_data_path}")
    raw = pd.read_csv(ingle_data_path)
    recs = []
    for r in raw.itertuples(index=False):
        fub = pd.to_numeric(getattr(r, "Fub"), errors="coerce")
        if pd.isna(fub) or not (0 <= fub <= 1):
            continue
        res = standardise_smiles(getattr(r, "SMILES"))
        if not res.ok or res.inchikey in ppbr_keys:
            continue
        scaf = bemis_murcko_scaffold(res.canonical_smiles)
        if scaf in forbidden_scaffolds:
            continue  # would leak into PPBR_AZ valid/test
        recs.append({"canonical_smiles": res.canonical_smiles, "inchikey": res.inchikey,
                     "scaffold": scaf, "ppb": 100.0 * (1.0 - float(fub))})
    df = pd.DataFrame(recs).drop_duplicates("inchikey").reset_index(drop=True)
    logger.info("Ingle scaffold-safe augmentation compounds: %d", len(df))
    return df


def _metrics(y_true, y_pred, mk):
    m = regression_metrics(y_true, y_pred, **mk)
    return {k: round(m[k], 4) for k in ("MAE", "RMSE", "R2", "Spearman", "high_binding_MAE", "fu_MAE")}


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    paths = Paths.create()
    logger = get_logger("augment", log_file="logs/train_augmented.log")

    cache = FeatureCache.load(paths.interim / "features.npz")
    split = pd.read_csv(paths.splits / "scaffold_split.csv")
    ppbr = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    ppbr_keys = set(ppbr["inchikey"].dropna())

    tr_idx = split.loc[split.split == "train", "row_index"].to_numpy()
    va_idx = split.loc[split.split == "valid", "row_index"].to_numpy()
    te_idx = split.loc[split.split == "test", "row_index"].to_numpy()
    train_scaffolds = split.loc[split.split == "train", "scaffold"].tolist()
    forbidden = set(split.loc[split.split.isin(["valid", "test"]), "scaffold"])

    ingle = _load_ingle_safe(cfg, paths, ppbr_keys, forbidden, logger)

    # Compute Ingle features.
    ingle_desc, _ = compute_descriptor_matrix(ingle["canonical_smiles"].tolist())
    ingle_morgan = morgan_matrix(ingle["canonical_smiles"].tolist(),
                                 cfg["features"]["morgan"]["radius"], cfg["features"]["morgan"]["n_bits"])

    # Assemble augmented training arrays.
    desc_raw_aug = np.vstack([cache.desc_raw[tr_idx], ingle_desc])
    morgan_aug = np.vstack([cache.morgan[tr_idx], ingle_morgan.astype(cache.morgan.dtype)])
    y_aug_pct = np.concatenate([cache.y_percent[tr_idx], ingle["ppb"].to_numpy()])
    # Empty/NaN scaffolds (acyclic molecules) become one explicit group so GroupKFold accepts them.
    groups_raw = train_scaffolds + ingle["scaffold"].tolist()
    groups_aug = np.array([g if isinstance(g, str) and g else "__acyclic__" for g in groups_raw],
                          dtype=object)
    logger.info("Augmented training set: %d PPBR_AZ + %d Ingle = %d",
                len(tr_idx), len(ingle), len(y_aug_pct))

    fcfg = cfg["features"]
    cleaner = DescriptorCleaner(fcfg["descriptor_variance_threshold"],
                                fcfg["descriptor_corr_threshold"]).fit(desc_raw_aug, cache.desc_names)

    def build(desc_raw, morgan, rep):
        d = cleaner.transform(desc_raw)
        return d if rep == "descriptors" else np.hstack([d, morgan])

    tf = TargetTransformer(cfg["target"]["transform"], float(cfg["target"]["epsilon"]),
                           tuple(cfg["target"]["clip_percent"]))
    y_aug_t = tf.forward(y_aug_pct)
    y_va, y_te = cache.y_percent[va_idx], cache.y_percent[te_idx]
    mk = dict(high_binding_threshold=cfg["evaluation"]["high_binding_threshold_percent"],
              bands=tuple(cfg["evaluation"]["binding_bands_percent"]))
    n_trials = min(cfg["tuning"]["n_trials"], 30)

    # Tune (grouped CV) + fit each member; collect valid/test predictions.
    tuned, va_logit, te_logit, member_test = {}, {}, {}, {}
    for key, mname, rep in MEMBERS:
        X_tr = build(desc_raw_aug, morgan_aug, rep)
        X_va = build(cache.desc_raw[va_idx], cache.morgan[va_idx], rep)
        X_te = build(cache.desc_raw[te_idx], cache.morgan[te_idx], rep)
        logger.info("Grouped-CV tuning %s (%s/%s, %d trials)...", key, mname, rep, n_trials)
        tr = tune_model_grouped(mname, X_tr, y_aug_t, y_aug_pct, groups_aug, tf,
                                n_splits=3, n_trials=n_trials, seed=seed)
        model = make_model(mname, seed=seed, **tr.best_params)
        model.fit(X_tr, y_aug_t)
        tuned[key] = {"best_params": tr.best_params, "cv_mae": round(tr.best_valid_mae, 4)}
        va_logit[key] = model.predict(X_va)
        te_logit[key] = model.predict(X_te)
        member_test[key] = _metrics(y_te, tf.inverse(te_logit[key]), mk)
        logger.info("  %s grouped-CV MAE=%.3f -> test MAE=%.3f", key, tr.best_valid_mae,
                    member_test[key]["MAE"])

    keys = [k for k, _, _ in MEMBERS]
    cons_va = tf.inverse(np.vstack([va_logit[k] for k in keys]).mean(0))
    cons_te = tf.inverse(np.vstack([te_logit[k] for k in keys]).mean(0))
    consensus_valid = regression_metrics(y_va, cons_va, **mk)
    consensus_test = _metrics(y_te, cons_te, mk)

    # Original (baseline) consensus test metrics for comparison.
    imp = pd.read_csv(paths.results / "improved_results.csv")
    orig = imp[imp.experiment_id == "T_consensus"].iloc[0]
    orig_metrics = {"MAE": round(float(orig["MAE"]), 4), "RMSE": round(float(orig["RMSE"]), 4),
                    "R2": round(float(orig["R2"]), 4), "Spearman": round(float(orig["Spearman"]), 4),
                    "high_binding_MAE": round(float(orig["high_binding_MAE"]), 4),
                    "fu_MAE": round(float(orig["fraction_unbound_MAE"]), 4)}

    summary = {
        "design": "augment train with scaffold-safe non-overlapping Ingle; grouped-CV tuning; "
                  "evaluate on unchanged PPBR_AZ scaffold test",
        "n_ppbr_train": int(len(tr_idx)), "n_ingle_added": int(len(ingle)),
        "n_augmented_train": int(len(y_aug_pct)), "n_test": int(len(te_idx)),
        "tuned": tuned, "member_test": member_test,
        "augmented_consensus_test": consensus_test,
        "original_consensus_test": orig_metrics,
        "delta_vs_original": {k: round(consensus_test[k] - orig_metrics[k], 4) for k in orig_metrics},
    }
    (paths.results / "augmented_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Augmentation + grouped-CV experiment (PPBR_AZ scaffold test) ===")
    print(f"Train: {len(tr_idx)} PPBR_AZ + {len(ingle)} Ingle = {len(y_aug_pct)}; test: {len(te_idx)}\n")
    hdr = f"{'metric':18s}{'original':>12s}{'augmented':>12s}{'delta':>10s}"
    print(hdr); print("-" * len(hdr))
    for k in ("MAE", "RMSE", "R2", "Spearman", "high_binding_MAE", "fu_MAE"):
        d = summary["delta_vs_original"][k]
        better = ("better" if ((k in ("R2", "Spearman") and d > 0) or
                               (k not in ("R2", "Spearman") and d < 0)) else "worse/eq")
        print(f"{k:18s}{orig_metrics[k]:12.4f}{consensus_test[k]:12.4f}{d:+10.4f}  {better}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Augmentation + grouped-CV improvement experiment.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
