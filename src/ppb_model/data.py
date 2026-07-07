"""Loading and filtering of the raw PPBR_AZ dataset.

This module only *reads* the raw file; it never modifies it. All cleaning happens
downstream in the prepare_data pipeline, which records every change in an audit table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import resolve_path


def load_raw(config: dict[str, Any]) -> pd.DataFrame:
    """Load the raw PPBR_AZ CSV as a DataFrame, validating the expected schema.

    Args:
        config: parsed config dict; uses config["data"] keys.

    Returns:
        The raw DataFrame with a stable integer ``record_id`` column added.

    Raises:
        FileNotFoundError: if the raw CSV is missing.
        KeyError: if any expected column is absent.
    """
    dcfg = config["data"]
    raw_path = resolve_path(dcfg["raw_csv"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    df = pd.read_csv(raw_path)
    expected = [dcfg["id_col"], dcfg["smiles_col"], dcfg["target_col"], dcfg["species_col"]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Raw dataset missing expected columns {missing}; found {list(df.columns)}")
    df = df.reset_index(drop=True)
    df.insert(0, "record_id", df.index.astype(int))
    return df


def filter_species(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Filter to the configured species (default: human). Returns a copy.

    If ``species_filter`` is null/empty, the frame is returned unchanged (all species).
    """
    dcfg = config["data"]
    species = dcfg.get("species_filter")
    if not species:
        return df.copy()
    sub = df[df[dcfg["species_col"]] == species].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for species={species!r}; available: "
            f"{sorted(df[dcfg['species_col']].unique())}"
        )
    return sub.reset_index(drop=True)


def load_clean_and_split(config: dict[str, Any], split_method: str | None = None) -> dict[str, Any]:
    """Load the cleaned modelling table joined to a saved split.

    Args:
        config: parsed config dict.
        split_method: "scaffold" (default from config) or "random".

    Returns:
        dict with keys: ``df`` (clean frame, split-ordered by row_index),
        ``smiles``, ``y_percent`` (np arrays), and ``idx`` mapping
        {"train","valid","test"} -> integer row-index arrays.

    Raises:
        FileNotFoundError: if the processed data or split file is missing.
    """
    method = split_method or config["split"]["method"]
    processed = resolve_path("data/processed/ppbr_az_human_clean.csv")
    split_path = resolve_path(f"data/splits/{method}_split.csv")
    if not processed.is_file():
        raise FileNotFoundError(f"Run Stage 2 first; missing {processed}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Run Stage 4-5 first; missing {split_path}")
    df = pd.read_csv(processed)
    split = pd.read_csv(split_path)
    idx = {name: split.loc[split["split"] == name, "row_index"].to_numpy(dtype=int)
           for name in ("train", "valid", "test")}
    return {
        "df": df,
        "smiles": df["canonical_smiles"].tolist(),
        "y_percent": df["y_percent"].to_numpy(dtype=float),
        "idx": idx,
        "split_method": method,
    }


def schema_report(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight schema/quality summary of a dataframe (no mutation)."""
    dcfg = config["data"]
    target = dcfg["target_col"]
    y = pd.to_numeric(df[target], errors="coerce")
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "n_missing_smiles": int(df[dcfg["smiles_col"]].isna().sum()),
        "n_missing_target": int(y.isna().sum()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_std": float(y.std()),
        "n_unique_smiles": int(df[dcfg["smiles_col"]].nunique()),
    }
