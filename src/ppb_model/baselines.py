"""Model factory for baseline and conventional regressors.

Keeps model construction in one place so every stage builds identically seeded models.
The proposed improved model (Stage 8) reuses these builders via the hybrid representation.
"""
from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRegressor

_BUILDERS = {
    "median": lambda seed, **kw: DummyRegressor(strategy="median"),
    "ridge": lambda seed, **kw: Ridge(alpha=kw.get("alpha", 1.0), random_state=seed),
    "elasticnet": lambda seed, **kw: ElasticNet(
        alpha=kw.get("alpha", 0.01), l1_ratio=kw.get("l1_ratio", 0.5),
        random_state=seed, max_iter=10000),
    "rf": lambda seed, **kw: RandomForestRegressor(
        n_estimators=kw.get("n_estimators", 400), max_depth=kw.get("max_depth", None),
        min_samples_leaf=kw.get("min_samples_leaf", 1), n_jobs=-1, random_state=seed),
    "extratrees": lambda seed, **kw: ExtraTreesRegressor(
        n_estimators=kw.get("n_estimators", 400), n_jobs=-1, random_state=seed),
    "hgb": lambda seed, **kw: HistGradientBoostingRegressor(
        learning_rate=kw.get("learning_rate", 0.1), max_depth=kw.get("max_depth", None),
        max_iter=kw.get("max_iter", 400), l2_regularization=kw.get("l2_regularization", 0.0),
        random_state=seed),
    "xgb": lambda seed, **kw: XGBRegressor(
        n_estimators=kw.get("n_estimators", 500), max_depth=kw.get("max_depth", 6),
        learning_rate=kw.get("learning_rate", 0.05), subsample=kw.get("subsample", 0.8),
        colsample_bytree=kw.get("colsample_bytree", 0.8),
        reg_lambda=kw.get("reg_lambda", 1.0), n_jobs=-1, random_state=seed,
        tree_method="hist"),
}


def available_models() -> list[str]:
    """Return the list of known model names."""
    return sorted(_BUILDERS)


def make_model(name: str, seed: int = 42, **params: Any):
    """Construct a regressor by name with a fixed seed and optional hyperparameters.

    Raises:
        KeyError: if the model name is unknown.
    """
    if name not in _BUILDERS:
        raise KeyError(f"Unknown model {name!r}; available: {available_models()}")
    return _BUILDERS[name](seed, **params)
