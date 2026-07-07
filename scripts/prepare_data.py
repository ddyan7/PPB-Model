"""Stage 2 entry point: inspect, standardise, and clean the PPBR_AZ dataset.

Produces (nothing in the raw file is modified):
    data/processed/ppbr_az_human_clean.csv  - cleaned, de-duplicated modelling table
    reports/tables/data_cleaning_audit.csv  - one row per raw record, every action logged
    reports/results/stage2_summary.json     - schema + cleaning summary

Cleaning policy (documented, never silent):
    * parse/standardisation failure         -> EXCLUDED
    * salt / multi-fragment parent taken     -> KEPT (parent structure), flagged
    * duplicate structure (same InChIKey),
      measurements agree (spread <= tol)     -> AGGREGATED to median, one row kept
    * conflicting measurements (spread > tol) -> AGGREGATED to median, flagged; kept unless
      spread exceeds hard_conflict_percent    -> then EXCLUDED as unreliable
    * ceiling-censored value (Y >= censor_high) -> KEPT, flagged (handled in Stage 3 target step)

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppb_model.data import filter_species, load_raw, schema_report
from ppb_model.standardisation import standardise_smiles
from ppb_model.utils import Paths, get_logger, load_config, resolve_path, set_seed

# Cleaning thresholds (in % bound units).
AGREE_TOL_PCT = 10.0        # replicate spread <= this is considered agreement
HARD_CONFLICT_PCT = 30.0    # spread above this -> structure excluded as unreliable
CENSOR_HIGH_PCT = 99.9      # ceiling-censoring flag


def _make_audit_row(record_id, original_smiles, standardised_smiles,
                    target_original, target_processed, issue, action, reason, included):
    return {
        "record_id": record_id,
        "original_smiles": original_smiles,
        "standardised_smiles": standardised_smiles,
        "target_original": target_original,
        "target_processed": target_processed,
        "issue_detected": issue,
        "action_taken": action,
        "reason": reason,
        "included_in_final_dataset": included,
    }


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    paths = Paths.create()
    logger = get_logger("stage2", log_file="logs/stage2_prepare_data.log")

    dcfg = cfg["data"]
    scfg = cfg["standardisation"]
    smiles_col, target_col, id_col = dcfg["smiles_col"], dcfg["target_col"], dcfg["id_col"]

    # ---- Load + filter -------------------------------------------------------
    raw = load_raw(cfg)
    logger.info("Loaded raw dataset: %d rows, %d cols", raw.shape[0], raw.shape[1])
    human = filter_species(raw, cfg)
    logger.info("Filtered to species=%r: %d rows", dcfg.get("species_filter"), len(human))
    pre = schema_report(human, cfg)
    logger.info("Human schema: mean=%.2f median=%.2f min=%.2f max=%.2f unique_smiles=%d",
                pre["target_mean"], pre["target_median"], pre["target_min"],
                pre["target_max"], pre["n_unique_smiles"])

    # ---- Standardise each record --------------------------------------------
    audit_rows: list[dict] = []
    std_records: list[dict] = []  # successfully standardised, pre-dedup

    for row in human.itertuples(index=False):
        rid = getattr(row, "record_id")
        smi = getattr(row, smiles_col)
        y_raw = getattr(row, target_col)
        y = pd.to_numeric(y_raw, errors="coerce")

        if pd.isna(y):
            audit_rows.append(_make_audit_row(rid, smi, None, y_raw, None,
                              "missing_target", "excluded", "target_not_numeric", False))
            continue

        res = standardise_smiles(
            smi,
            strip_salts=scfg["strip_salts"],
            neutralise_charges=scfg["neutralise_charges"],
            keep_stereo=scfg["keep_stereo"],
        )
        if not res.ok:
            audit_rows.append(_make_audit_row(rid, smi, None, y_raw, None,
                              "standardisation_failed", "excluded", res.error, False))
            continue

        std_records.append({
            "record_id": rid,
            "drug_id": getattr(row, id_col),
            "original_smiles": smi,
            "canonical_smiles": res.canonical_smiles,
            "inchikey": res.inchikey,
            "y_percent": float(y),
            "parent_multi_fragment": res.parent_multi_fragment,
        })

    std_df = pd.DataFrame(std_records)
    logger.info("Standardised OK: %d / %d human records", len(std_df), len(human))

    # ---- Duplicate / conflict resolution by InChIKey -------------------------
    final_rows: list[dict] = []
    for key, grp in std_df.groupby("inchikey", sort=False):
        grp = grp.sort_values("record_id")
        vals = grp["y_percent"].to_numpy()
        spread = float(vals.max() - vals.min())
        rep = grp.iloc[0]  # representative record for identity/SMILES
        y_agg = float(np.median(vals))
        n = len(grp)
        multi = bool(grp["parent_multi_fragment"].any())

        # Base issue/reason for the salt/mixture note.
        salt_note = "salt_or_mixture_parent_taken" if multi else None

        if n == 1:
            issue = salt_note or "none"
            action = "kept"
            reason = salt_note or "clean_single_measurement"
            included = True
        elif spread <= AGREE_TOL_PCT:
            issue = "duplicate_structure"
            action = "aggregated_median"
            reason = f"n={n}, spread={spread:.2f}pct<=tol; " + (salt_note or "agreeing_replicates")
            included = True
        elif spread <= HARD_CONFLICT_PCT:
            issue = "conflicting_measurement"
            action = "aggregated_median_flagged"
            reason = f"n={n}, spread={spread:.2f}pct (soft conflict); " + (salt_note or "kept_as_median")
            included = True
        else:
            issue = "conflicting_measurement"
            action = "excluded"
            reason = f"n={n}, spread={spread:.2f}pct>{HARD_CONFLICT_PCT} hard conflict; unreliable"
            included = False

        # Censoring flag (does not change inclusion; handled in Stage 3).
        censored = y_agg >= CENSOR_HIGH_PCT
        if censored and included:
            issue = (issue + ";ceiling_censored") if issue != "none" else "ceiling_censored"
            reason = reason + f"; y_agg={y_agg:.2f}>=censor_high"

        # One audit row per raw record in the group.
        for _, r in grp.iterrows():
            audit_rows.append(_make_audit_row(
                int(r["record_id"]), r["original_smiles"], r["canonical_smiles"],
                float(r["y_percent"]), y_agg if included else None,
                issue, action, reason, included))

        if included:
            final_rows.append({
                "record_id": int(rep["record_id"]),
                "drug_id": rep["drug_id"],
                "canonical_smiles": rep["canonical_smiles"],
                "inchikey": key,
                "y_percent": y_agg,
                "n_measurements": n,
                "measurement_spread_pct": spread,
                "salt_or_mixture": multi,
                "ceiling_censored": bool(censored),
            })

    final_df = pd.DataFrame(final_rows).sort_values("record_id").reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows).sort_values("record_id").reset_index(drop=True)

    # ---- Persist -------------------------------------------------------------
    processed_path = paths.processed / "ppbr_az_human_clean.csv"
    audit_path = paths.tables / "data_cleaning_audit.csv"
    summary_path = paths.results / "stage2_summary.json"
    final_df.to_csv(processed_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "human_rows": int(len(human)),
        "standardised_ok": int(len(std_df)),
        "unique_structures_inchikey": int(std_df["inchikey"].nunique()) if len(std_df) else 0,
        "final_rows": int(len(final_df)),
        "excluded_total": int((~audit_df["included_in_final_dataset"]).groupby(audit_df["record_id"]).first().sum()),
        "n_salt_or_mixture": int(final_df["salt_or_mixture"].sum()) if len(final_df) else 0,
        "n_ceiling_censored": int(final_df["ceiling_censored"].sum()) if len(final_df) else 0,
        "n_conflicts_soft": int(
            ((final_df["n_measurements"] > 1) & (final_df["measurement_spread_pct"] > AGREE_TOL_PCT)).sum()
        ) if len(final_df) else 0,
        "final_target_summary": {
            "mean": float(final_df["y_percent"].mean()),
            "median": float(final_df["y_percent"].median()),
            "std": float(final_df["y_percent"].std()),
            "min": float(final_df["y_percent"].min()),
            "max": float(final_df["y_percent"].max()),
            "pct_ge_90": float((final_df["y_percent"] >= 90).mean() * 100),
        },
        "thresholds": {
            "agree_tol_pct": AGREE_TOL_PCT,
            "hard_conflict_pct": HARD_CONFLICT_PCT,
            "censor_high_pct": CENSOR_HIGH_PCT,
        },
        "outputs": {
            "processed_csv": str(processed_path),
            "audit_csv": str(audit_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Final modelling rows: %d (from %d human)", len(final_df), len(human))
    logger.info("Wrote: %s", processed_path)
    logger.info("Wrote: %s", audit_path)
    logger.info("Wrote: %s", summary_path)
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: prepare and clean PPBR_AZ data.")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
