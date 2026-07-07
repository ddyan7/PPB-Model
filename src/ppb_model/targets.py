"""PPB target transformations with invertible forward/inverse mappings.

The raw target is **percent bound** ``Y`` in [0, 100]. Models are trained in a
transformed space and predictions are mapped back to percent for reporting, so every
transformation provides an exact inverse.

Fraction conventions:
    fb = Y / 100            fraction bound
    fu = 1 - fb             fraction unbound

Boundary handling: values are clipped to an assay-plausible percent window
(``clip_percent``) and then guarded by ``epsilon`` in fraction space, so that
log/logit transforms never see exactly 0 or 1.

Supported methods:
    none              t = Y (percent)
    fraction_bound    t = fb
    fraction_unbound  t = fu
    log_fu            t = log10(fu)                    (Watanabe-style)
    logit             t = ln(fb / fu)                  (Han-style LogIt)
    lnKa              t = 0.5 * ln(fb / fu)            (pseudo-equilibrium constant)
    clipped           t = fb clipped to [eps, 1-eps]
"""
from __future__ import annotations

from typing import Any

import numpy as np

_METHODS = {"none", "fraction_bound", "fraction_unbound", "log_fu", "logit", "lnKa", "clipped"}


class TargetTransformer:
    """Invertible transformer between percent-bound and a modelling target space."""

    def __init__(self, method: str = "logit", epsilon: float = 1e-3,
                 clip_percent: tuple[float, float] = (0.1, 99.9)) -> None:
        if method not in _METHODS:
            raise ValueError(f"Unknown target method {method!r}; choose from {sorted(_METHODS)}")
        if not 0 < epsilon < 0.5:
            raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")
        lo, hi = clip_percent
        if not 0 <= lo < hi <= 100:
            raise ValueError(f"clip_percent must satisfy 0<=lo<hi<=100, got {clip_percent}")
        self.method = method
        self.epsilon = float(epsilon)
        self.clip_percent = (float(lo), float(hi))

    # -- helpers ---------------------------------------------------------------
    def _to_fb(self, y_percent: np.ndarray) -> np.ndarray:
        """Percent -> clipped, epsilon-guarded fraction bound."""
        y = np.asarray(y_percent, dtype=float)
        lo, hi = self.clip_percent
        y = np.clip(y, lo, hi)
        fb = y / 100.0
        return np.clip(fb, self.epsilon, 1.0 - self.epsilon)

    # -- forward ---------------------------------------------------------------
    def forward(self, y_percent: np.ndarray) -> np.ndarray:
        """Map percent-bound to the modelling target space."""
        y = np.asarray(y_percent, dtype=float)
        if self.method == "none":
            return y.copy()
        fb = self._to_fb(y)
        fu = 1.0 - fb
        if self.method == "fraction_bound":
            return fb
        if self.method == "fraction_unbound":
            return fu
        if self.method == "log_fu":
            return np.log10(fu)
        if self.method == "logit":
            return np.log(fb / fu)
        if self.method == "lnKa":
            return 0.5 * np.log(fb / fu)
        if self.method == "clipped":
            return fb
        raise AssertionError(self.method)  # pragma: no cover

    # -- inverse ---------------------------------------------------------------
    def inverse(self, t: np.ndarray) -> np.ndarray:
        """Map a modelling-space value/prediction back to percent-bound."""
        t = np.asarray(t, dtype=float)
        if self.method == "none":
            return t.copy()
        if self.method in ("fraction_bound", "clipped"):
            fb = t
        elif self.method == "fraction_unbound":
            fb = 1.0 - t
        elif self.method == "log_fu":
            fb = 1.0 - np.power(10.0, t)
        elif self.method == "logit":
            fb = 1.0 / (1.0 + np.exp(-t))
        elif self.method == "lnKa":
            fb = 1.0 / (1.0 + np.exp(-2.0 * t))
        else:  # pragma: no cover
            raise AssertionError(self.method)
        return np.clip(fb, 0.0, 1.0) * 100.0


def make_transformer(config: dict[str, Any]) -> TargetTransformer:
    """Build a :class:`TargetTransformer` from the ``target`` block of the config."""
    tcfg = config["target"]
    return TargetTransformer(
        method=tcfg.get("transform", "logit"),
        epsilon=float(tcfg.get("epsilon", 1e-3)),
        clip_percent=tuple(tcfg.get("clip_percent", (0.1, 99.9))),
    )


def fraction_unbound_from_percent(y_percent: np.ndarray) -> np.ndarray:
    """Convenience: percent bound -> fraction unbound (no clipping), for fu-space error metrics."""
    return 1.0 - np.asarray(y_percent, dtype=float) / 100.0
