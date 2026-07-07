"""Tests for target transformations: invertibility, monotonicity, boundary safety."""
import numpy as np
import pytest

from ppb_model.targets import TargetTransformer, fraction_unbound_from_percent


@pytest.mark.parametrize("method", ["none", "fraction_bound", "fraction_unbound",
                                    "log_fu", "logit", "lnKa", "clipped"])
def test_roundtrip_invertible(method):
    tf = TargetTransformer(method=method, epsilon=1e-3, clip_percent=(0.1, 99.9))
    y = np.array([11.18, 50.0, 88.07, 95.43, 99.9])
    y_clipped = np.clip(y, 0.1, 99.9)
    recovered = tf.inverse(tf.forward(y))
    assert np.allclose(recovered, y_clipped, atol=1e-6)


def test_logit_monotonic():
    tf = TargetTransformer(method="logit")
    y = np.array([10.0, 50.0, 90.0, 99.0])
    t = tf.forward(y)
    assert np.all(np.diff(t) > 0)  # strictly increasing


def test_boundary_no_nonfinite():
    tf = TargetTransformer(method="logit")
    t = tf.forward(np.array([0.0, 100.0]))  # extremes clipped, not inf
    assert np.all(np.isfinite(t))


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        TargetTransformer(method="not_a_method")


def test_fraction_unbound_helper():
    assert np.allclose(fraction_unbound_from_percent(np.array([99.0])), np.array([0.01]))
