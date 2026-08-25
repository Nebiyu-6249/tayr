"""Kalman filter tests."""

from __future__ import annotations

import numpy as np
import pytest

from tayr.errors import GeometryError
from tayr.geometry import xyxy_to_cxcywh
from tayr.tracking.kalman import KalmanBoxFilter


def box_at(cx: float, cy: float, side: float = 20.0) -> list[float]:
    h = side / 2
    return [cx - h, cy - h, cx + h, cy + h]


class TestInitialisation:
    def test_initial_estimate_matches_the_first_box(self) -> None:
        kf = KalmanBoxFilter(box_at(100, 200))
        assert np.allclose(kf.box_xyxy, box_at(100, 200))

    def test_initial_velocity_is_zero(self) -> None:
        assert np.allclose(KalmanBoxFilter(box_at(0, 0)).velocity, [0.0, 0.0])

    @pytest.mark.parametrize(
        "kwargs",
        [{"process_noise_scale": 0.0}, {"measurement_noise_scale": -1.0}, {"dt": 0.0}],
    )
    def test_invalid_parameters_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(GeometryError):
            KalmanBoxFilter(box_at(0, 0), **kwargs)

    def test_multiple_boxes_rejected(self) -> None:
        with pytest.raises(GeometryError, match="single box"):
            KalmanBoxFilter([box_at(0, 0), box_at(10, 10)])


class TestTracking:
    def test_learns_constant_velocity(self) -> None:
        """Fed a target moving 5px/frame right, the filter's velocity estimate must
        converge on that. Without it, prediction cannot help association."""
        kf = KalmanBoxFilter(box_at(0, 100))
        for i in range(1, 30):
            kf.predict()
            kf.update(box_at(5 * i, 100))
        vx, vy = kf.velocity
        assert vx == pytest.approx(5.0, abs=0.5)
        assert vy == pytest.approx(0.0, abs=0.5)

    def test_prediction_leads_the_target(self) -> None:
        kf = KalmanBoxFilter(box_at(0, 100))
        for i in range(1, 20):
            kf.predict()
            kf.update(box_at(5 * i, 100))
        predicted = xyxy_to_cxcywh(kf.predict())
        # After the 19th observation at x=95, one step ahead should be near x=100.
        assert predicted[0] == pytest.approx(100.0, abs=2.0)

    def test_smooths_measurement_noise(self) -> None:
        """The filtered path must vary less than the noisy measurements it was fed."""
        rng = np.random.default_rng(1337)
        kf = KalmanBoxFilter(box_at(0, 100))
        measured, filtered = [], []
        for i in range(1, 60):
            true_x = 2.0 * i
            noisy_x = true_x + rng.normal(0, 3.0)
            kf.predict()
            kf.update(box_at(noisy_x, 100))
            measured.append(noisy_x - true_x)
            filtered.append(float(xyxy_to_cxcywh(kf.box_xyxy)[0]) - true_x)
        assert np.std(filtered) < np.std(measured)

    def test_size_is_tracked(self) -> None:
        kf = KalmanBoxFilter(box_at(0, 0, side=10))
        for _ in range(20):
            kf.predict()
            kf.update(box_at(0, 0, side=40))
        w = xyxy_to_cxcywh(kf.box_xyxy)[2]
        assert w == pytest.approx(40.0, abs=2.0)

    def test_repeated_prediction_never_yields_a_degenerate_box(self) -> None:
        """Long occlusions drive the state far from any measurement; the reported box
        must stay valid or every downstream geometry call would raise."""
        kf = KalmanBoxFilter(box_at(100, 100))
        for _ in range(500):
            b = kf.predict()
            assert b[2] > b[0] and b[3] > b[1]

    def test_covariance_stays_positive_definite(self) -> None:
        """Joseph-form update: the shorter (I-KH)P form can drift negative over a long
        track, and a negative covariance makes the gain meaningless."""
        kf = KalmanBoxFilter(box_at(0, 0))
        for i in range(1, 400):
            kf.predict()
            kf.update(box_at(i * 0.5, i * 0.25))
        eigenvalues = np.linalg.eigvalsh((kf.P + kf.P.T) / 2)
        assert eigenvalues.min() > -1e-9, f"min eigenvalue {eigenvalues.min()}"
