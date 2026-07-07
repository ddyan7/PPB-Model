"""Core training harness: assemble representations, fit in transformed target space,
inverse-transform predictions, and evaluate in percent space.

The single entry point :func:`run_experiment` guarantees leakage safety:
    * descriptor cleaning/scaling is fitted on training rows only;
    * the target transform is fitted from config (fixed, not data-tuned);
    * validation is used for model selection, the test set only for final reporting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .baselines import make_model
from .evaluation import regression_metrics
from .features import DescriptorCleaner
from .targets import TargetTransformer


@dataclass
class FeatureCache:
    """In-memory view of the cached raw features (see scripts/build_features.py)."""

    desc_raw: np.ndarray
    desc_names: list[str]
    morgan: np.ndarray
    maccs: np.ndarray
    y_percent: np.ndarray

    @classmethod
    def load(cls, npz_path) -> "FeatureCache":
        d = np.load(npz_path, allow_pickle=True)
        return cls(
            desc_raw=d["desc_raw"],
            desc_names=list(d["desc_names"]),
            morgan=d["morgan"].astype(np.float32),
            maccs=d["maccs"].astype(np.float32),
            y_percent=d["y_percent"].astype(float),
        )


def assemble_representation(
    representation: str,
    cache: FeatureCache,
    idx: dict[str, np.ndarray],
    variance_threshold: float,
    corr_threshold: float,
) -> tuple[dict[str, np.ndarray], int, list[str]]:
    """Build train/valid/test feature matrices for a representation (train-only cleaning).

    Returns (matrices, n_features, feature_names).
    """
    tr, va, te = idx["train"], idx["valid"], idx["test"]

    def _clean_descriptors():
        cleaner = DescriptorCleaner(variance_threshold, corr_threshold)
        cleaner.fit(cache.desc_raw[tr], cache.desc_names)
        return ({s: cleaner.transform(cache.desc_raw[i]) for s, i in
                 (("train", tr), ("valid", va), ("test", te))}, cleaner.feature_names_)

    if representation == "descriptors":
        mats, names = _clean_descriptors()
    elif representation == "morgan":
        mats = {"train": cache.morgan[tr], "valid": cache.morgan[va], "test": cache.morgan[te]}
        names = [f"morgan_{i}" for i in range(cache.morgan.shape[1])]
    elif representation == "maccs":
        mats = {"train": cache.maccs[tr], "valid": cache.maccs[va], "test": cache.maccs[te]}
        names = [f"maccs_{i}" for i in range(cache.maccs.shape[1])]
    elif representation == "hybrid":
        dmats, dnames = _clean_descriptors()
        mats = {s: np.hstack([dmats[s], {"train": cache.morgan[tr], "valid": cache.morgan[va],
                                          "test": cache.morgan[te]}[s]]) for s in ("train", "valid", "test")}
        names = dnames + [f"morgan_{i}" for i in range(cache.morgan.shape[1])]
    else:
        raise ValueError(f"Unknown representation {representation!r}")
    n_features = mats["train"].shape[1]
    return mats, n_features, names


@dataclass
class ExperimentResult:
    """Outcome of one fit/evaluate experiment."""

    meta: dict[str, Any]
    valid_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_pred_pct: np.ndarray
    valid_pred_pct: np.ndarray
    model: Any
    fitted_transformer: TargetTransformer


def run_experiment(
    *,
    model_name: str,
    representation: str,
    transform_method: str,
    cache: FeatureCache,
    idx: dict[str, np.ndarray],
    config: dict[str, Any],
    model_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> ExperimentResult:
    """Fit one model on one representation with one target transform; evaluate in %.

    The model trains on transformed targets; predictions are inverse-transformed to
    percent before metrics are computed.
    """
    seed = seed if seed is not None else config["project"]["seed"]
    model_params = model_params or {}
    fcfg = config["features"]
    ecfg = config["evaluation"]

    mats, n_features, _names = assemble_representation(
        representation, cache, idx,
        fcfg["descriptor_variance_threshold"], fcfg["descriptor_corr_threshold"])

    tf = TargetTransformer(method=transform_method,
                           epsilon=float(config["target"]["epsilon"]),
                           clip_percent=tuple(config["target"]["clip_percent"]))
    y_pct = cache.y_percent
    y_tr = tf.forward(y_pct[idx["train"]])

    model = make_model(model_name, seed=seed, **model_params)
    t0 = time.time()
    model.fit(mats["train"], y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    test_pred_pct = tf.inverse(model.predict(mats["test"]))
    predict_time = time.time() - t0
    valid_pred_pct = tf.inverse(model.predict(mats["valid"]))

    kwargs = dict(high_binding_threshold=ecfg["high_binding_threshold_percent"],
                  bands=tuple(ecfg["binding_bands_percent"]))
    test_metrics = regression_metrics(y_pct[idx["test"]], test_pred_pct, **kwargs)
    valid_metrics = regression_metrics(y_pct[idx["valid"]], valid_pred_pct, **kwargs)

    meta = {
        "model": model_name,
        "representation": representation,
        "target_transformation": transform_method,
        "seed": seed,
        "n_features": int(n_features),
        "train_size": int(len(idx["train"])),
        "valid_size": int(len(idx["valid"])),
        "test_size": int(len(idx["test"])),
        "train_time_s": round(train_time, 3),
        "predict_time_s": round(predict_time, 4),
        "model_params": model_params,
    }
    return ExperimentResult(meta, valid_metrics, test_metrics, test_pred_pct,
                            valid_pred_pct, model, tf)
