"""Experiment results table: flatten results to the project schema and persist CSV/MD/JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .train import ExperimentResult

# Column order per the project brief.
COLUMNS = [
    "experiment_id", "dataset_version", "target_definition", "target_transformation",
    "representation", "model", "split_method", "random_seed",
    "training_size", "validation_size", "test_size", "number_of_features",
    "hyperparameter_method", "MAE", "RMSE", "R2", "Spearman", "Pearson",
    "median_absolute_error", "high_binding_MAE", "fraction_unbound_MAE",
    "prediction_interval_coverage", "training_time", "prediction_time", "notes",
]


def result_row(
    result: ExperimentResult,
    *,
    experiment_id: str,
    split_method: str,
    dataset_version: str = "ppbr_az_human_clean_v1",
    hyperparameter_method: str = "defaults",
    prediction_interval_coverage: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build one results-table row from an :class:`ExperimentResult` (test-set metrics)."""
    m, meta = result.test_metrics, result.meta
    return {
        "experiment_id": experiment_id,
        "dataset_version": dataset_version,
        "target_definition": "percent_bound",
        "target_transformation": meta["target_transformation"],
        "representation": meta["representation"],
        "model": meta["model"],
        "split_method": split_method,
        "random_seed": meta["seed"],
        "training_size": meta["train_size"],
        "validation_size": meta["valid_size"],
        "test_size": meta["test_size"],
        "number_of_features": meta["n_features"],
        "hyperparameter_method": hyperparameter_method,
        "MAE": round(m["MAE"], 4),
        "RMSE": round(m["RMSE"], 4),
        "R2": round(m["R2"], 4),
        "Spearman": round(m["Spearman"], 4),
        "Pearson": round(m["Pearson"], 4),
        "median_absolute_error": round(m["MedAE"], 4),
        "high_binding_MAE": round(m["high_binding_MAE"], 4),
        "fraction_unbound_MAE": round(m["fu_MAE"], 5),
        "prediction_interval_coverage": (round(prediction_interval_coverage, 4)
                                         if prediction_interval_coverage is not None else None),
        "training_time": meta["train_time_s"],
        "prediction_time": meta["predict_time_s"],
        "notes": notes or f"valid_MAE={result.valid_metrics['MAE']:.3f};"
                          f"fu_GMFE_high={m['fu_GMFE_high']:.3f}",
    }


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no external deps)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_results_table(rows: list[dict[str, Any]], results_dir: Path, stem: str) -> pd.DataFrame:
    """Persist rows as CSV, Markdown, and JSON; return the DataFrame."""
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / f"{stem}.csv", index=False)
    (results_dir / f"{stem}.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    (results_dir / f"{stem}.md").write_text(_to_markdown(df), encoding="utf-8")
    return df
