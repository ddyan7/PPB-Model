"""Head-to-head: published Ingle et al. 2016 models vs this model on identical held-out data.

ppb_usable_dataset.csv carries Ingle's own model predictions (kNN/SVM/RF/consensus, in
fraction-unbound space) and a Set column. We evaluate every model on the SAME molecules:
Ingle's held-out sets (Dte, T1, T2) with PPBR_AZ InChIKey overlaps removed, so the compounds
are out-of-sample for Ingle's models AND unseen by ours.

Important asymmetry (stated, not hidden): on this data Ingle's models are tested
*in-distribution* (same source/assay they were built on), while ours is a *cross-source
transfer* model trained on AstraZeneca data. Ingle therefore has a home-field advantage; the
question answered is how a transfer model compares to a purpose-built in-distribution model.

Outputs:
    reports/results/head_to_head_ingle.csv / .json

Usage:
    python scripts/head_to_head.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ppb_model.evaluation import regression_metrics
from ppb_model.predict import load_bundle, predict_smiles
from ppb_model.results import _to_markdown
from ppb_model.standardisation import standardise_smiles
from ppb_model.utils import Paths, get_logger, load_config, resolve_path, set_seed

HELD_OUT_SETS = {"Dte", "T1", "T2"}          # Ingle's out-of-sample partitions
PUBLISHED = {"Ingle_kNN": "kNN_Pred", "Ingle_SVM": "SVM_Pred",
             "Ingle_RF": "RF_Pred", "Ingle_consensus": "Con_Pred"}


def _metric_row(label, y_true, y_pred, mk, n_total):
    m = regression_metrics(y_true, y_pred, **mk)
    return {"model": label, "n": m["n"], "MAE": round(m["MAE"], 3), "RMSE": round(m["RMSE"], 3),
            "R2": round(m["R2"], 3), "Spearman": round(m["Spearman"], 3),
            "high_binding_MAE": round(m["high_binding_MAE"], 3), "fu_MAE": round(m["fu_MAE"], 4)}


def run(config_path: str, data_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("head2head", log_file="logs/head_to_head.log")

    raw = pd.read_csv(resolve_path(data_path))
    held = raw[raw["Set"].isin(HELD_OUT_SETS)].copy()
    logger.info("Ingle held-out rows (Dte+T1+T2): %d", len(held))

    ppbr_keys = set(pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")["inchikey"].dropna())

    rows = []
    for r in held.itertuples(index=False):
        fub = pd.to_numeric(getattr(r, "Fub"), errors="coerce")
        con = pd.to_numeric(getattr(r, "Con_Pred"), errors="coerce")
        if pd.isna(fub) or not (0 <= fub <= 1) or pd.isna(con):
            continue
        res = standardise_smiles(getattr(r, "SMILES"))
        if not res.ok:
            continue
        rec = {"inchikey": res.inchikey, "canonical_smiles": res.canonical_smiles,
               "set": getattr(r, "Set"), "obs_ppb": 100.0 * (1.0 - float(fub))}
        for label, col in PUBLISHED.items():
            v = pd.to_numeric(getattr(r, col), errors="coerce")
            rec[label] = 100.0 * (1.0 - float(v)) if pd.notna(v) else np.nan
        rows.append(rec)

    common = pd.DataFrame(rows)
    n_before = len(common)
    common = common[~common["inchikey"].isin(ppbr_keys)].drop_duplicates("inchikey").reset_index(drop=True)
    logger.info("Common held-out set after overlap removal: %d (removed %d)",
                len(common), n_before - len(common))

    y_true = common["obs_ppb"].to_numpy(dtype=float)
    ecfg = cfg["evaluation"]
    mk = dict(high_binding_threshold=ecfg["high_binding_threshold_percent"],
              bands=tuple(ecfg["binding_bands_percent"]))

    # My models on the identical rows.
    for name, bundle_file in (("Mine_consensus", "final_consensus.joblib"),
                              ("Mine_xgb_hybrid", "final_xgb_hybrid.joblib")):
        bpath = paths.models / bundle_file
        if bpath.is_file():
            preds = predict_smiles(load_bundle(bpath), common["canonical_smiles"].tolist())
            common[name] = preds["predicted_ppb"].to_numpy(dtype=float)

    table = []
    for label in list(PUBLISHED) + ["Mine_consensus", "Mine_xgb_hybrid"]:
        if label in common.columns:
            table.append(_metric_row(label, y_true, common[label].to_numpy(dtype=float), mk, len(common)))
    res_df = pd.DataFrame(table).sort_values("MAE").reset_index(drop=True)

    res_df.to_csv(paths.results / "head_to_head_ingle.csv", index=False)
    (paths.results / "head_to_head_ingle.md").write_text(_to_markdown(res_df), encoding="utf-8")
    (paths.results / "head_to_head_ingle.json").write_text(
        json.dumps({"n_common_heldout": int(len(common)),
                    "held_out_sets": sorted(HELD_OUT_SETS),
                    "results": res_df.to_dict("records")}, indent=2), encoding="utf-8")

    print(f"Identical held-out compounds evaluated: {len(common)} "
          f"(Ingle Dte+T1+T2 minus PPBR_AZ overlaps)\n")
    print(res_df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Head-to-head vs Ingle published models.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="PPB_Datasets/ppb_usable_dataset.csv")
    args = ap.parse_args()
    run(args.config, args.data)


if __name__ == "__main__":
    main()
