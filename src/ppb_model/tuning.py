"""Reproducible hyperparameter optimisation with Optuna.

Tuning uses the training set to fit and the **validation** set to score (percent-space
MAE after inverse-transforming the logit prediction). The test set is never touched here.
Feature matrices are assembled once and reused across trials for speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna

from sklearn.model_selection import GroupKFold

from .baselines import make_model
from .targets import TargetTransformer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest a hyperparameter dict for the given model."""
    if model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    if model_name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }
    if model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
    raise ValueError(f"No search space defined for model {model_name!r}")


@dataclass
class TuningResult:
    """Best hyperparameters and metadata from a tuning run."""

    model_name: str
    best_params: dict[str, Any]
    best_valid_mae: float
    n_trials: int
    search_space: list[str]
    seed: int


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train_transformed: np.ndarray,
    X_valid: np.ndarray,
    y_valid_percent: np.ndarray,
    transformer: TargetTransformer,
    *,
    n_trials: int = 50,
    seed: int = 42,
) -> TuningResult:
    """Optimise a model's hyperparameters against validation percent-space MAE."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, model_name)
        model = make_model(model_name, seed=seed, **params)
        model.fit(X_train, y_train_transformed)
        pred_pct = transformer.inverse(model.predict(X_valid))
        return float(np.mean(np.abs(y_valid_percent - pred_pct)))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # Record the search-space parameter names from the best trial.
    space = list(study.best_trial.params.keys())
    return TuningResult(model_name, study.best_params, float(study.best_value),
                        n_trials, space, seed)


def tune_model_grouped(
    model_name: str,
    X: np.ndarray,
    y_transformed: np.ndarray,
    y_percent: np.ndarray,
    groups: np.ndarray,
    transformer: TargetTransformer,
    *,
    n_splits: int = 3,
    n_trials: int = 30,
    seed: int = 42,
) -> TuningResult:
    """Optimise hyperparameters against **scaffold-grouped** CV percent-space MAE.

    Groups (Bemis-Murcko scaffolds) are kept intact within a fold, so the CV selection
    signal reflects generalisation to unseen scaffolds rather than a single noisy fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y_percent, groups))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, model_name)
        maes = []
        for tr_i, va_i in folds:
            model = make_model(model_name, seed=seed, **params)
            model.fit(X[tr_i], y_transformed[tr_i])
            pred = transformer.inverse(model.predict(X[va_i]))
            maes.append(float(np.mean(np.abs(y_percent[va_i] - pred))))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuningResult(model_name, study.best_params, float(study.best_value),
                        n_trials, list(study.best_trial.params.keys()), seed)
