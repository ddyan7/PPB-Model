"""External validation on the Ingle et al. 2016 dataset (ppb_usable_dataset.csv).

This is a genuine out-of-distribution test: the model was trained only on human PPBR_AZ.
To keep it leakage-free we remove any Ingle compound whose standardised InChIKey also
appears anywhere in the human PPBR_AZ set, then score the frozen model(s) on the remainder.

Steps:
    1. Load Ingle data; identify SMILES + Fub (fraction unbound).
    2. Standardise -> canonical SMILES + InChIKey (drop parse failures).
    3. Drop InChIKey overlaps with PPBR_AZ human (report the count).
    4. Convert Fub -> percent bound: PPB% = 100 * (1 - Fub).
    5. Predict with the saved bundle(s); score overall and in-applicability-domain.

Outputs:
    reports/results/external_validation_ingle.json
    reports/results/external_validation_predictions.csv

Usage:
    python scripts/external_validation.py --config configs/default.yaml \
        --data "data/raw/ppb_usable_dataset.csv"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ppb_model.evaluation import regression_metrics
from ppb_model.predict import load_bundle, predict_smiles
from ppb_model.standardisation import standardise_smiles
from ppb_model.utils import Paths, get_logger, load_config, resolve_path, set_seed


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in columns {list(df.columns)}")


def run(config_path: str, data_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("external", log_file="logs/external_validation.log")

    raw = pd.read_csv(resolve_path(data_path))
    smiles_col = _find_col(raw, ["SMILES", "smiles", "Drug"])
    fub_col = _find_col(raw, ["Fub", "fub", "FUB"])
    logger.info("Loaded Ingle data: %d rows (SMILES=%r, Fub=%r)", len(raw), smiles_col, fub_col)

    # PPBR_AZ human InChIKeys (everything the model could have seen).
    ppbr = pd.read_csv(paths.processed / "ppbr_az_human_clean.csv")
    ppbr_keys = set(ppbr["inchikey"].dropna())

    # Standardise + convert target, track exclusions.
    records, n_parse_fail, n_bad_target = [], 0, 0
    for row in raw.itertuples(index=False):
        smi = getattr(row, smiles_col)
        fub = pd.to_numeric(getattr(row, fub_col), errors="coerce")
        if pd.isna(fub) or not (0.0 <= fub <= 1.0):
            n_bad_target += 1
            continue
        res = standardise_smiles(smi)
        if not res.ok:
            n_parse_fail += 1
            continue
        records.append({"orig_smiles": smi, "canonical_smiles": res.canonical_smiles,
                        "inchikey": res.inchikey, "ppb_percent": 100.0 * (1.0 - float(fub))})

    ext = pd.DataFrame(records)
    n_std = len(ext)
    overlap_mask = ext["inchikey"].isin(ppbr_keys)
    n_overlap = int(overlap_mask.sum())
    ext = ext[~overlap_mask].drop_duplicates("inchikey").reset_index(drop=True)
    logger.info("Standardised %d; %d parse fails; %d bad targets; %d PPBR_AZ overlaps removed; "
                "%d unique external compounds retained",
                n_std, n_parse_fail, n_bad_target, n_overlap, len(ext))

    y_true = ext["ppb_percent"].to_numpy(dtype=float)
    ecfg = cfg["evaluation"]
    mk = dict(high_binding_threshold=ecfg["high_binding_threshold_percent"],
              bands=tuple(ecfg["binding_bands_percent"]))

    summary = {
        "source": str(resolve_path(data_path)),
        "n_raw": int(len(raw)),
        "n_standardised": n_std,
        "n_parse_failures": n_parse_fail,
        "n_bad_target": n_bad_target,
        "n_ppbr_az_overlap_removed": n_overlap,
        "n_external_evaluated": int(len(ext)),
        "models": {},
    }

    bundles = {"consensus": paths.models / "final_consensus.joblib",
               "xgb_hybrid": paths.models / "final_xgb_hybrid.joblib"}
    pred_out = ext[["orig_smiles", "canonical_smiles", "ppb_percent"]].copy()

    for name, bpath in bundles.items():
        if not bpath.is_file():
            continue
        preds = predict_smiles(load_bundle(bpath), ext["canonical_smiles"].tolist())
        y_pred = preds["predicted_ppb"].to_numpy(dtype=float)
        in_ad = preds["in_applicability_domain"].to_numpy(dtype=bool)
        pred_out[f"pred_{name}"] = y_pred
        pred_out[f"in_ad_{name}"] = in_ad

        m_all = regression_metrics(y_true, y_pred, **mk)
        m_ad = regression_metrics(y_true[in_ad], y_pred[in_ad], **mk) if in_ad.any() else {}
        summary["models"][name] = {
            "all": {k: round(v, 4) for k, v in m_all.items() if isinstance(v, (int, float))},
            "in_domain_only": {k: round(v, 4) for k, v in m_ad.items() if isinstance(v, (int, float))},
            "in_domain_fraction": round(float(in_ad.mean()), 4),
        }
        logger.info("%s: all MAE=%.2f R2=%.3f Spearman=%.3f | in-AD(%.0f%%) MAE=%.2f",
                    name, m_all["MAE"], m_all["R2"], m_all["Spearman"],
                    100 * in_ad.mean(), m_ad.get("MAE", float("nan")))

    pred_out.to_csv(paths.results / "external_validation_predictions.csv", index=False)
    (paths.results / "external_validation_ingle.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="External validation on the Ingle dataset.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/raw/ppb_usable_dataset.csv")
    args = ap.parse_args()
    run(args.config, args.data)


if __name__ == "__main__":
    main()
