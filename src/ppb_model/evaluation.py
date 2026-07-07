"""Evaluation metrics for PPB regression.

All metrics are computed in **percent-bound space** (predictions are inverse-transformed
from the modelling target before scoring), so results are directly comparable across
target transformations and to the literature.

Beyond standard regression metrics we report:
    high_binding_mae   - MAE on compounds with true PPB >= high_binding_threshold (%)
    fu_gmfe_high       - geometric-mean fold-error of fraction unbound on the high-binding
                         subset; captures relative error where 1% at 99% doubles fu
    band_mae           - MAE within literature-motivated binding bands
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def regression_metrics(
    y_true_pct: np.ndarray,
    y_pred_pct: np.ndarray,
    high_binding_threshold: float = 90.0,
    bands: tuple[float, ...] = (50, 80, 90, 95, 99),
) -> dict[str, Any]:
    """Compute the full metric suite in percent space.

    Args:
        y_true_pct, y_pred_pct: true/predicted PPB in percent (0-100).
        high_binding_threshold: percent cutoff defining the high-binding subset.
        bands: ascending band edges (percent) for stratified MAE.
    """
    y_true = np.asarray(y_true_pct, dtype=float)
    y_pred = np.asarray(y_pred_pct, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    abs_err = np.abs(y_true - y_pred)

    # Correlations are undefined when predictions are constant (e.g. the median baseline).
    constant_pred = np.std(y_pred) == 0 or np.std(y_true) == 0
    spearman = float("nan") if constant_pred else float(stats.spearmanr(y_true, y_pred).statistic)
    pearson = float("nan") if constant_pred else float(stats.pearsonr(y_true, y_pred).statistic)

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": _rmse(y_true, y_pred),
        "R2": _r2(y_true, y_pred),
        "Spearman": spearman,
        "Pearson": pearson,
        "MedAE": float(np.median(abs_err)),
    }

    # High-binding subset MAE (decisive co-metric).
    hi = y_true >= high_binding_threshold
    out["high_binding_MAE"] = float(np.mean(abs_err[hi])) if hi.any() else float("nan")
    out["high_binding_n"] = int(hi.sum())

    # Fraction-unbound relative error on the high-binding subset.
    fu_true = np.clip((100.0 - y_true) / 100.0, 1e-4, 1.0)
    fu_pred = np.clip((100.0 - y_pred) / 100.0, 1e-4, 1.0)
    out["fu_MAE"] = float(np.mean(np.abs(fu_true - fu_pred)))
    if hi.any():
        fold = np.maximum(fu_pred[hi] / fu_true[hi], fu_true[hi] / fu_pred[hi])
        out["fu_GMFE_high"] = float(np.exp(np.mean(np.log(fold))))
    else:
        out["fu_GMFE_high"] = float("nan")

    # Banded MAE.
    edges = [0.0, *bands, 100.0001]
    band_mae = {}
    for lo, hi_e in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & (y_true < hi_e)
        label = f"[{lo:g},{hi_e if hi_e <= 100 else 100:g})"
        band_mae[label] = {"n": int(m.sum()),
                           "MAE": float(np.mean(abs_err[m])) if m.any() else float("nan")}
    out["band_MAE"] = band_mae
    return out
