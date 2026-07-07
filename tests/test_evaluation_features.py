"""Tests for evaluation metrics and feature cleaning."""
import numpy as np

from ppb_model.evaluation import regression_metrics
from ppb_model.features import DescriptorCleaner, maccs_matrix, morgan_matrix


def test_perfect_prediction():
    y = np.array([20.0, 50.0, 90.0, 99.0])
    m = regression_metrics(y, y.copy())
    assert m["MAE"] == 0.0
    assert abs(m["R2"] - 1.0) < 1e-9


def test_high_binding_subset_selected():
    y_true = np.array([10.0, 85.0, 92.0, 98.0])
    y_pred = np.array([10.0, 85.0, 90.0, 96.0])
    m = regression_metrics(y_true, y_pred, high_binding_threshold=90.0)
    assert m["high_binding_n"] == 2  # 92 and 98


def test_descriptor_cleaner_train_only_and_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    X[:, 0] = 1.0                 # constant column -> dropped
    X[:, 2] = X[:, 1]            # duplicate -> one dropped by correlation filter
    X[3, 4] = np.nan             # missing -> imputed
    names = [f"f{i}" for i in range(5)]
    cleaner = DescriptorCleaner(variance_threshold=1e-8, corr_threshold=0.95).fit(X, names)
    Xt = cleaner.transform(X)
    assert np.isfinite(Xt).all()
    assert Xt.shape[1] < 5         # some columns removed
    assert "f0" not in cleaner.feature_names_  # constant dropped


def test_fingerprint_shapes_and_binary():
    smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
    morgan = morgan_matrix(smiles, radius=2, n_bits=1024)
    maccs = maccs_matrix(smiles)
    assert morgan.shape == (3, 1024)
    assert maccs.shape == (3, 167)
    assert set(np.unique(morgan)).issubset({0, 1})
