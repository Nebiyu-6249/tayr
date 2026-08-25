"""Constant-velocity Kalman filter for a bounding box.

Written directly rather than taken from `filterpy`, whose last release was 1.4.5 on
2018-10-10 `[VERIFIED: https://pypi.org/pypi/filterpy/json]`. The filter is short, and
writing it means Q and R can be tuned for aerial motion rather than inherited from a
pedestrian-tracking default.

STATE: `[cx, cy, w, h, vx, vy]` - centre, size, and centre velocity.

Size carries no velocity term, deliberately. At 10 pixels on target a +/-1px measurement
error is 10% noise, and a size-velocity estimated from that is dominated by measurement
noise rather than by real scale change. Size is therefore observed and smoothed but never
extrapolated. Scale change rate - which the motion classifier needs - is computed from
the observed track history in `features.py`, where its noise is visible rather than
hidden inside a filter state.

MEASUREMENT: `[cx, cy, w, h]`, i.e. a detection converted from xyxy.

SCOPE: `predict()` advances the state one frame, for data association only. There is no
multi-step extrapolation here and none may be added - see the package docstring.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tayr.errors import GeometryError
from tayr.geometry import FloatArray, cxcywh_to_xyxy, xyxy_to_cxcywh

_STATE_DIM = 6
_MEAS_DIM = 4


class KalmanBoxFilter:
    """Tracks one box through time.

    `process_noise_scale` (Q) governs how much the filter trusts its motion model;
    `measurement_noise_scale` (R) how much it trusts the detector. Aerial targets change
    direction faster than pedestrians, so Q defaults higher than a typical SORT setting.
    Both are tunable because the right values are an empirical question this project has
    not yet answered on real data.
    """

    def __init__(
        self,
        initial_box_xyxy: npt.ArrayLike,
        *,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        if process_noise_scale <= 0 or measurement_noise_scale <= 0:
            raise GeometryError(
                f"noise scales must be positive; got Q={process_noise_scale} "
                f"R={measurement_noise_scale}"
            )
        if dt <= 0:
            raise GeometryError(f"dt must be positive; got {dt}")

        measurement = xyxy_to_cxcywh(initial_box_xyxy).reshape(-1)
        if measurement.shape[0] != _MEAS_DIM:
            raise GeometryError(f"expected a single box, got shape {measurement.shape}")

        # State transition: centre integrates velocity; size persists.
        self._F = np.eye(_STATE_DIM, dtype=np.float64)
        self._F[0, 4] = dt
        self._F[1, 5] = dt

        # Observation: we measure centre and size, never velocity.
        self._H = np.zeros((_MEAS_DIM, _STATE_DIM), dtype=np.float64)
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = self._H[3, 3] = 1.0

        self._Q = np.eye(_STATE_DIM, dtype=np.float64) * process_noise_scale
        # Velocity is the least constrained part of the state, so it accumulates the
        # most process noise.
        self._Q[4:, 4:] *= 10.0
        self._R = np.eye(_MEAS_DIM, dtype=np.float64) * measurement_noise_scale

        self.x = np.zeros(_STATE_DIM, dtype=np.float64)
        self.x[:4] = measurement
        # Initial velocity is unknown, so its covariance starts large: the filter should
        # believe the first few measurements over its own (absent) motion estimate.
        self.P = np.eye(_STATE_DIM, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0

    def predict(self) -> FloatArray:
        """Advance one frame and return the predicted box as xyxy."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        # A predicted box can drift to non-positive size; clamp only the returned value,
        # never the state, so the filter can recover when a measurement arrives.
        return self._to_xyxy(self.x)

    def update(self, box_xyxy: npt.ArrayLike) -> None:
        """Correct the state with a measured box."""
        z = xyxy_to_cxcywh(box_xyxy).reshape(-1)
        if z.shape[0] != _MEAS_DIM:
            raise GeometryError(f"expected a single box, got shape {z.shape}")

        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        identity = np.eye(_STATE_DIM, dtype=np.float64)
        # Joseph form: stays positive-definite under floating-point error, where the
        # shorter (I - KH)P form can drift negative over a long track.
        factor = identity - K @ self._H
        self.P = factor @ self.P @ factor.T + K @ self._R @ K.T

    @property
    def box_xyxy(self) -> FloatArray:
        """Current filtered box estimate."""
        return self._to_xyxy(self.x)

    @property
    def velocity(self) -> FloatArray:
        """Centre velocity `[vx, vy]` in pixels per frame."""
        return self.x[4:6].copy()

    @staticmethod
    def _to_xyxy(state: FloatArray) -> FloatArray:
        cx, cy, w, h = state[:4]
        return cxcywh_to_xyxy([cx, cy, max(w, 1e-6), max(h, 1e-6)])
