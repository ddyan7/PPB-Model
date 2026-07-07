"""Load a saved model bundle and predict PPB for new SMILES.

A bundle is a self-contained dict holding everything needed for inference:
    kind          "single" or "consensus"
    model/members the fitted regressor(s) (trained on the logit target)
    representation/member_reps  which feature set each model consumes
    cleaner       the train-fitted DescriptorCleaner
    transformer   the TargetTransformer (logit) for inverse-mapping to percent
    morgan        {"radius", "n_bits"} fingerprint settings
    morgan_train  training fingerprints (for applicability-domain similarity)
    ad_threshold  Tanimoto cutoff defining the applicability domain

This closes the loop from raw SMILES to a percent-bound prediction with an
applicability-domain flag and (for consensus) an ensemble-uncertainty score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import compute_descriptor_matrix, morgan_matrix
from .standardisation import standardise_smiles
from .uncertainty import applicability_domain_flag, ensemble_uncertainty, max_train_tanimoto


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a model bundle saved by scripts/finalize_models.py."""
    bundle = joblib.load(path)
    required = {"kind", "cleaner", "transformer", "morgan", "morgan_train", "ad_threshold"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Bundle missing keys {missing}")
    return bundle


def _features_for(rep: str, desc_clean: np.ndarray, morgan: np.ndarray) -> np.ndarray:
    if rep == "descriptors":
        return desc_clean
    if rep == "hybrid":
        return np.hstack([desc_clean, morgan])
    raise ValueError(f"Unsupported representation for inference: {rep!r}")


def predict_smiles(bundle: dict[str, Any], smiles: list[str]) -> pd.DataFrame:
    """Predict PPB (%) for a list of SMILES using a loaded bundle.

    Returns a DataFrame with columns: input_smiles, standardised_smiles,
    predicted_ppb, predicted_fraction_unbound, max_training_similarity,
    in_applicability_domain, uncertainty_score (NaN for single-model bundles),
    and a status column noting any standardisation failures.
    """
    if not isinstance(smiles, list) or not smiles:
        raise ValueError("smiles must be a non-empty list of strings")

    std, status = [], []
    for s in smiles:
        res = standardise_smiles(s)
        std.append(res.canonical_smiles if res.ok else None)
        status.append("ok" if res.ok else (res.error or "failed"))

    valid_mask = np.array([s is not None for s in std])
    valid_smiles = [s for s in std if s is not None]

    n = len(smiles)
    pred_pct = np.full(n, np.nan)
    max_sim = np.full(n, np.nan)
    unc = np.full(n, np.nan)

    if valid_smiles:
        desc_raw, _ = compute_descriptor_matrix(valid_smiles)
        desc_clean = bundle["cleaner"].transform(desc_raw)
        morgan = morgan_matrix(valid_smiles, bundle["morgan"]["radius"], bundle["morgan"]["n_bits"])
        tf = bundle["transformer"]

        if bundle["kind"] == "single":
            X = _features_for(bundle["representation"], desc_clean, morgan)
            preds = tf.inverse(bundle["model"].predict(X))
        elif bundle["kind"] == "consensus":
            member_preds = []
            for name, model in bundle["members"].items():
                rep = bundle["member_reps"][name]
                X = _features_for(rep, desc_clean, morgan)
                member_preds.append(tf.inverse(model.predict(X)))
            member_preds = np.vstack(member_preds)
            preds = member_preds.mean(axis=0)
            unc[valid_mask] = ensemble_uncertainty(member_preds)
        else:
            raise ValueError(f"Unknown bundle kind {bundle['kind']!r}")

        sims = max_train_tanimoto(morgan, bundle["morgan_train"])
        pred_pct[valid_mask] = preds
        max_sim[valid_mask] = sims

    in_ad = applicability_domain_flag(max_sim, bundle["ad_threshold"])
    in_ad = np.where(np.isnan(max_sim), False, in_ad)

    return pd.DataFrame({
        "input_smiles": smiles,
        "standardised_smiles": std,
        "predicted_ppb": pred_pct,
        "predicted_fraction_unbound": (100 - pred_pct) / 100,
        "max_training_similarity": max_sim,
        "in_applicability_domain": in_ad,
        "uncertainty_score": unc,
        "status": status,
    })
