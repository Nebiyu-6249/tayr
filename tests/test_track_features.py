"""Motion feature tests.

Each test builds a synthetic trajectory that HAS the property being measured, and
asserts the corresponding feature responds. This checks the arithmetic is right - it is
NOT evidence that these features separate real drones from real birds. No such data is
in hand (docs/RESEARCH.md 14.4), and any claim about real separability must wait for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tayr.errors import GeometryError
from tayr.tracking.features import MIN_OBSERVATIONS, TrackFeatures, extract_features

FPS = 30.0


def track_from_centres(
    centres: np.ndarray, side: float | np.ndarray = 20.0, frames: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build xyxy boxes and frame indices from a centre path."""
    n = len(centres)
    sides = np.full(n, side, dtype=np.float64) if np.isscalar(side) else np.asarray(side)
    half = sides / 2.0
    boxes = np.stack(
        [
            centres[:, 0] - half,
            centres[:, 1] - half,
            centres[:, 0] + half,
            centres[:, 1] + half,
        ],
        axis=-1,
    )
    idx = np.arange(n, dtype=np.int64) if frames is None else np.asarray(frames, dtype=np.int64)
    return boxes, idx


def straight_line(n: int = 60, speed: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64)
    return track_from_centres(np.stack([t * speed, np.full(n, 300.0)], axis=-1))


def flapping(
    n: int = 120, hz: float = 5.0, amplitude: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """Steady forward motion with a vertical oscillation - a crude flapping proxy."""
    t = np.arange(n, dtype=np.float64)
    y = 300.0 + amplitude * np.sin(2 * np.pi * hz * t / FPS)
    return track_from_centres(np.stack([t * 3.0, y], axis=-1))


def hovering(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    jitter = rng.normal(0, 0.05, size=(n, 2))
    return track_from_centres(np.stack([np.full(n, 500.0), np.full(n, 300.0)], axis=-1) + jitter)


class TestGuards:
    def test_short_track_raises_rather_than_returning_zeros(self) -> None:
        boxes, frames = straight_line(n=MIN_OBSERVATIONS - 1)
        with pytest.raises(GeometryError, match="at least"):
            extract_features(boxes, frames)

    def test_mismatched_lengths(self) -> None:
        boxes, frames = straight_line(n=20)
        with pytest.raises(GeometryError, match="frame index"):
            extract_features(boxes, frames[:-1])

    def test_non_increasing_frames(self) -> None:
        boxes, frames = straight_line(n=20)
        bad = frames.copy()
        bad[5] = bad[4]
        with pytest.raises(GeometryError, match="strictly increasing"):
            extract_features(boxes, bad)

    def test_invalid_fps(self) -> None:
        boxes, frames = straight_line(n=20)
        with pytest.raises(GeometryError, match="fps"):
            extract_features(boxes, frames, fps=0.0)


class TestSpeed:
    def test_mean_speed_matches_construction(self) -> None:
        boxes, frames = straight_line(n=40, speed=4.0)
        assert extract_features(boxes, frames, fps=FPS).mean_speed == pytest.approx(4.0)

    def test_constant_speed_has_near_zero_variance(self) -> None:
        boxes, frames = straight_line(n=40, speed=4.0)
        assert extract_features(boxes, frames, fps=FPS).speed_variance == pytest.approx(
            0.0, abs=1e-9
        )

    def test_gaps_do_not_register_as_speed_spikes(self) -> None:
        """A track lost for ten frames then re-acquired has moved ten frames' worth.
        Dividing by the real frame delta is what keeps that from reading as a jump."""
        t = np.concatenate([np.arange(10), np.arange(20, 30)]).astype(np.int64)
        centres = np.stack([t * 4.0, np.full(len(t), 300.0)], axis=-1)
        boxes, _ = track_from_centres(centres, frames=list(t))
        f = extract_features(boxes, t, fps=FPS)
        assert f.mean_speed == pytest.approx(4.0)
        assert f.speed_variance == pytest.approx(0.0, abs=1e-9)


class TestOscillation:
    def test_recovers_the_construction_frequency(self) -> None:
        boxes, frames = flapping(n=150, hz=5.0)
        f = extract_features(boxes, frames, fps=FPS)
        assert f.vertical_oscillation_hz == pytest.approx(5.0, abs=0.5)
        assert f.vertical_oscillation_power > 0.5

    @pytest.mark.parametrize("hz", [2.0, 4.0, 8.0])
    def test_tracks_different_frequencies(self, hz: float) -> None:
        boxes, frames = flapping(n=200, hz=hz)
        assert extract_features(boxes, frames, fps=FPS).vertical_oscillation_hz == pytest.approx(
            hz, abs=0.6
        )

    def test_smooth_motion_has_low_oscillation_power(self) -> None:
        """A steady climb must not read as an oscillation - that is what detrending is
        for. Without it, every ascending target would look like it was flapping."""
        n = 100
        t = np.arange(n, dtype=np.float64)
        boxes, frames = track_from_centres(np.stack([t * 3.0, 300.0 - t * 2.0], axis=-1))
        f = extract_features(boxes, frames, fps=FPS)
        assert f.vertical_oscillation_power < 0.5

    def test_oscillating_beats_straight_on_power(self) -> None:
        osc = extract_features(*flapping(n=150, hz=5.0), fps=FPS)
        straight = extract_features(*straight_line(n=150), fps=FPS)
        assert osc.vertical_oscillation_power > straight.vertical_oscillation_power


class TestSmoothnessAndEntropy:
    def test_straight_path_has_smoothness_near_one(self) -> None:
        f = extract_features(*straight_line(n=50), fps=FPS)
        assert f.trajectory_smoothness == pytest.approx(1.0, abs=0.01)

    def test_wandering_path_is_less_smooth(self) -> None:
        rng = np.random.default_rng(3)
        n = 80
        steps = rng.normal(0, 6.0, size=(n, 2))
        centres = np.cumsum(steps, axis=0) + 400.0
        f = extract_features(*track_from_centres(centres), fps=FPS)
        assert f.trajectory_smoothness > 2.0

    def test_straight_path_has_low_heading_entropy(self) -> None:
        assert extract_features(*straight_line(n=50), fps=FPS).heading_entropy == pytest.approx(
            0.0, abs=0.01
        )

    def test_random_walk_has_high_heading_entropy(self) -> None:
        rng = np.random.default_rng(11)
        centres = np.cumsum(rng.normal(0, 6.0, size=(200, 2)), axis=0) + 400.0
        f = extract_features(*track_from_centres(centres), fps=FPS)
        assert f.heading_entropy > 0.8

    def test_entropy_is_normalised_to_unit_interval(self) -> None:
        rng = np.random.default_rng(5)
        centres = np.cumsum(rng.normal(0, 6.0, size=(300, 2)), axis=0)
        f = extract_features(*track_from_centres(centres), fps=FPS)
        assert 0.0 <= f.heading_entropy <= 1.0


class TestHoverAndScale:
    def test_hovering_target_registers_hover_fraction(self) -> None:
        f = extract_features(*hovering(n=60), fps=FPS)
        assert f.hover_fraction > 0.9

    def test_transiting_target_does_not(self) -> None:
        assert extract_features(*straight_line(n=50, speed=4.0), fps=FPS).hover_fraction == 0.0

    def test_constant_size_has_near_zero_scale_change(self) -> None:
        boxes, frames = straight_line(n=40)
        assert extract_features(boxes, frames, fps=FPS).scale_change_rate == pytest.approx(
            0.0, abs=1e-9
        )

    def test_approaching_target_registers_scale_change(self) -> None:
        """A target growing 3% per frame - the proxy for closing range."""
        n = 50
        t = np.arange(n, dtype=np.float64)
        sides = 10.0 * (1.03**t)
        centres = np.stack([np.full(n, 400.0), np.full(n, 300.0)], axis=-1)
        boxes, frames = track_from_centres(centres, side=sides)
        f = extract_features(boxes, frames, fps=FPS)
        assert f.scale_change_rate == pytest.approx(0.03, abs=0.005)


class TestVector:
    def test_vector_matches_the_declared_names(self) -> None:
        f = extract_features(*straight_line(n=40), fps=FPS)
        assert len(f.as_vector()) == len(TrackFeatures.feature_names())

    def test_vector_is_finite(self) -> None:
        """A NaN reaching the classifier would fail silently in some tree libraries."""
        for boxes, frames in (straight_line(50), flapping(120), hovering(60)):
            assert np.all(np.isfinite(extract_features(boxes, frames, fps=FPS).as_vector()))

    def test_returning_to_origin_does_not_divide_by_zero(self) -> None:
        """A closed loop has zero net displacement; smoothness must stay finite."""
        t = np.linspace(0, 2 * np.pi, 60)
        centres = np.stack([400 + 50 * np.cos(t), 300 + 50 * np.sin(t)], axis=-1)
        f = extract_features(*track_from_centres(centres), fps=FPS)
        assert np.isfinite(f.trajectory_smoothness)
        assert f.trajectory_smoothness > 10.0


class TestSeparationOnSyntheticTrajectories:
    def test_synthetic_flapping_and_hovering_differ_across_features(self) -> None:
        """Sanity check that the feature set distinguishes two constructed archetypes.

        This says the arithmetic works. It says NOTHING about real birds and drones -
        these trajectories were built to differ. Real separability is unmeasured.
        """
        bird_like = extract_features(*flapping(n=150, hz=5.0), fps=FPS)
        drone_like = extract_features(*hovering(n=150), fps=FPS)
        assert bird_like.vertical_oscillation_power > drone_like.vertical_oscillation_power
        assert bird_like.mean_speed > drone_like.mean_speed
        assert drone_like.hover_fraction > bird_like.hover_fraction
