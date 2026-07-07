"""Applicability domain (AD) and prediction uncertainty.

AD is estimated from structural similarity to the training set (maximum Tanimoto
similarity over Morgan bits): a low maximum similarity means the query is far from
anything seen in training and its prediction is extrapolative.

Uncertainty is estimated from disagreement among the members of a consensus ensemble
(standard deviation of member predictions in percent space), which the modelling plan
identifies as a lightweight, model-agnostic reliability signal.
"""
from __future__ import annotations

import numpy as np


def max_train_tanimoto(morgan_query: np.ndarray, morgan_train: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
    """Maximum Tanimoto similarity of each query molecule to any training molecule.

    Args:
        morgan_query: (n_query, n_bits) binary Morgan fingerprints.
        morgan_train: (n_train, n_bits) binary Morgan fingerprints.
        batch_size: query rows processed per matrix multiply (memory control).

    Returns:
        (n_query,) array of maximum Tanimoto similarities in [0, 1].
    """
    q = (morgan_query > 0).astype(np.float32)
    t = (morgan_train > 0).astype(np.float32)
    t_sum = t.sum(axis=1)  # popcount per training molecule
    out = np.empty(len(q), dtype=np.float32)
    for start in range(0, len(q), batch_size):
        qb = q[start:start + batch_size]
        inter = qb @ t.T                       # intersection popcounts
        qb_sum = qb.sum(axis=1, keepdims=True)
        union = qb_sum + t_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[start:start + batch_size] = sim.max(axis=1)
    return out


def applicability_domain_flag(max_sim: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean in-domain flag: True where max training similarity >= threshold."""
    return max_sim >= threshold


def ensemble_uncertainty(member_predictions_pct: np.ndarray) -> np.ndarray:
    """Per-sample uncertainty = std of member predictions (percent space).

    Args:
        member_predictions_pct: (n_members, n_samples) predictions in percent.

    Returns:
        (n_samples,) standard deviation across members.
    """
    arr = np.asarray(member_predictions_pct, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (members, samples), got shape {arr.shape}")
    return arr.std(axis=0, ddof=0)
