"""Promote the augmented model to final: rebuild deployable bundles on the augmented data.

Training set = PPBR_AZ scaffold-train (seed 42) + scaffold-safe non-overlapping Ingle
compounds, with the augmented-tuned hyperparameters. Rebuilds:
    models/final_consensus.joblib   (augmented; originals backed up as *_ppbr_only.joblib)
    models/final_xgb_hybrid.joblib
    reports/results/predictions_final.csv        (augmented model on PPBR_AZ splits)
    reports/results/augmented_final_summary.json (training composition + metrics + provenance)

Honesty note recorded in the summary: because Ingle is now training data, it can no longer
serve as an external validation set for the deployed model.

Usage:
    python scripts/promote_augmented.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from ppb_model.baselines import make_model
from ppb_model.evaluation import regression_metrics
from ppb_model.features import DescriptorCleaner, compute_descriptor_matrix, morgan_matrix
from ppb_model.predict import load_bundle, predict_smiles
from ppb_model.splitting import bemis_murcko_scaffold, scaffold_split
from ppb_model.standardisation import standardise_smiles
from ppb_model.targets import TargetTransformer
from ppb_model.train import FeatureCache
from ppb_model.uncertainty import max_train_tanimoto
from ppb_model.utils import Paths, get_logger, load_config, resolve_path, set_seed

MEMBER_REPS = {"xgb_hybrid": "hybrid", "rf_desc": "descriptors", "hgb_desc": "descriptors"}


def _self_sim_threshold(morgan_train, pct=5, batch=256):
    t = (morgan_train > 0).astype(np.float32)
    t_sum = t.sum(1)
    n = len(t)
    out = np.empty(n, np.float32)
    for s in range(0, n, batch):
        qb = t[s:s + batch]
        inter = qb @ t.T
        union = qb.sum(1, keepdims=True) + t_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        for r in range(qb.shape[0]):
            sim[r, s + r] = -1.0
        out[s:s + qb.shape[0]] = sim.max(1)
    return float(np.percentile(out, pct))


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    paths = Paths.create()
    logger = get_logger("promote", log_file="logs/promote_augmented.log")

    cache = FeatureCache.load(paths.interim / "features.npz")
    ppbr = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    smiles = ppbr["canonical_smiles"].tolist()
    fcfg, tcfg = cfg["features"], cfg["target"]
    tf = TargetTransformer(tcfg["transform"], float(tcfg["epsilon"]), tuple(tcfg["clip_percent"]))
    m_r, m_b = fcfg["morgan"]["radius"], fcfg["morgan"]["n_bits"]

    sp = scaffold_split(smiles, cfg["split"]["train_frac"], cfg["split"]["valid_frac"],
                        cfg["split"]["test_frac"], seed=seed)
    tr, va, te = sp["train"], sp["valid"], sp["test"]
    forbidden = {bemis_murcko_scaffold(smiles[i]) or "__acyclic__" for i in np.concatenate([va, te])}

    # Non-overlapping, scaffold-safe Ingle compounds + features.
    dcfg = cfg["data"]
    ingle_data_path = resolve_path(dcfg["ingle_data_csv"])
    if not ingle_data_path.is_file():
        raise FileNotFoundError(f"Ingle dataset not found: {ingle_data_path}")
    raw = pd.read_csv(ingle_data_path)

    ppbr_keys = set(ppbr["inchikey"].dropna())
    recs = []
    for r in raw.itertuples(index=False):
        fub = pd.to_numeric(getattr(r, "Fub"), errors="coerce")
        if pd.isna(fub) or not (0 <= fub <= 1):
            continue
        res = standardise_smiles(getattr(r, "SMILES"))
        if not res.ok or res.inchikey in ppbr_keys:
            continue
        scaf = bemis_murcko_scaffold(res.canonical_smiles) or "__acyclic__"
        if scaf in forbidden:
            continue
        recs.append({"smiles": res.canonical_smiles, "ppb": 100.0 * (1.0 - float(fub))})
    ing = pd.DataFrame(recs).reset_index(drop=True)
    logger.info("Scaffold-safe Ingle added to training: %d", len(ing))
    ing_desc, _ = compute_descriptor_matrix(ing["smiles"].tolist())
    ing_morgan = morgan_matrix(ing["smiles"].tolist(), m_r, m_b).astype(cache.morgan.dtype)

    # Augmented training arrays.
    desc_tr = np.vstack([cache.desc_raw[tr], ing_desc])
    morgan_tr = np.vstack([cache.morgan[tr], ing_morgan])
    y_tr_pct = np.concatenate([cache.y_percent[tr], ing["ppb"].to_numpy()])
    y_tr_t = tf.forward(y_tr_pct)

    cleaner = DescriptorCleaner(fcfg["descriptor_variance_threshold"],
                                fcfg["descriptor_corr_threshold"]).fit(desc_tr, cache.desc_names)
    aug_params = {k: v["best_params"] for k, v in
                  json.loads((paths.results / "augmented_results.json").read_text())["tuned"].items()}

    members = {}
    for key, rep in MEMBER_REPS.items():
        mname = {"xgb_hybrid": "xgb", "rf_desc": "rf", "hgb_desc": "hgb"}[key]
        X = cleaner.transform(desc_tr)
        if rep == "hybrid":
            X = np.hstack([X, morgan_tr])
        model = make_model(mname, seed=seed, **aug_params[key])
        model.fit(X, y_tr_t)
        members[key] = model

    ad_threshold = _self_sim_threshold(morgan_tr, pct=5)
    common = {"cleaner": cleaner, "transformer": tf,
              "morgan": {"radius": m_r, "n_bits": m_b},
              "morgan_train": morgan_tr.astype(np.uint8), "ad_threshold": ad_threshold}

    # Back up the original (PPBR_AZ-only) bundles, then write augmented bundles.
    for fn in ("final_consensus.joblib", "final_xgb_hybrid.joblib"):
        src = paths.models / fn
        if src.is_file():
            shutil.copy2(src, paths.models / fn.replace(".joblib", "_ppbr_only.joblib"))
    joblib.dump({"kind": "consensus", "members": members, "member_reps": MEMBER_REPS,
                 "selected_id": "augmented_consensus", **common},
                paths.models / "final_consensus.joblib", compress=3)
    joblib.dump({"kind": "single", "model": members["xgb_hybrid"], "representation": "hybrid", **common},
                paths.models / "final_xgb_hybrid.joblib", compress=3)
    logger.info("Wrote augmented bundles (originals backed up as *_ppbr_only.joblib); AD=%.3f", ad_threshold)

    # Final predictions on PPBR_AZ splits using the deployed (augmented) consensus.
    split_df = pd.read_csv(paths.splits / "scaffold_split.csv")
    scaffold_map = dict(zip(split_df["row_index"], split_df["scaffold"]))
    bundle = load_bundle(paths.models / "final_consensus.joblib")
    mk = dict(high_binding_threshold=cfg["evaluation"]["high_binding_threshold_percent"],
              bands=tuple(cfg["evaluation"]["binding_bands_percent"]))
    rows, test_metrics = [], None
    for name, idx in (("train", tr), ("valid", va), ("test", te)):
        sm = [smiles[i] for i in idx]
        preds = predict_smiles(bundle, sm)
        obs = cache.y_percent[idx]
        pp = preds["predicted_ppb"].to_numpy()
        if name == "test":
            test_metrics = {k: round(regression_metrics(obs, pp, **mk)[k], 4)
                            for k in ("MAE", "RMSE", "R2", "Spearman", "high_binding_MAE", "fu_MAE")}
        for j, i in enumerate(idx):
            rows.append({"compound_id": ppbr.iloc[i]["drug_id"], "canonical_smiles": smiles[i],
                         "data_split": name, "observed_ppb": float(obs[j]),
                         "predicted_ppb": float(pp[j]),
                         "observed_fraction_unbound": float((100 - obs[j]) / 100),
                         "predicted_fraction_unbound": float((100 - pp[j]) / 100),
                         "absolute_error": float(abs(obs[j] - pp[j])),
                         "scaffold": scaffold_map.get(int(i), ""),
                         "maximum_training_similarity": float(preds["max_training_similarity"].iloc[j]),
                         "uncertainty_score": float(preds["uncertainty_score"].iloc[j]),
                         "in_applicability_domain": bool(preds["in_applicability_domain"].iloc[j]),
                         "model_name": "augmented_consensus"})
    pd.DataFrame(rows).to_csv(paths.results / "predictions_final.csv", index=False)

    summary = {
        "deployed_model": "augmented_consensus",
        "training_composition": {"ppbr_az_train": int(len(tr)), "ingle_scaffold_safe": int(len(ing)),
                                 "total": int(len(y_tr_pct))},
        "test_metrics_ppbr_scaffold": test_metrics,
        "ad_threshold": round(ad_threshold, 4),
        "external_validation_note": ("Ingle (ppb_usable) is now TRAINING data and can no longer serve "
                                     "as external validation for the deployed model; a fresh third-party "
                                     "set would be required. The prior Ingle external-validation and "
                                     "head-to-head results describe the earlier PPBR_AZ-only model, "
                                     "preserved as *_ppbr_only.joblib."),
        "robustness_ref": "reports/results/confirm_robustness.json",
    }
    (paths.results / "augmented_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Promoted augmented model. Test MAE=%.3f", test_metrics["MAE"])
    print(json.dumps(summary, indent=2))
    demo = predict_smiles(bundle, ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"])
    print("\nSmoke test:", [round(v, 1) for v in demo["predicted_ppb"]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote the augmented model to final.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
