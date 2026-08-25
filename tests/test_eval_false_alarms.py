"""False-alarm-rate tests."""

from __future__ import annotations

import numpy as np
import pytest

from tayr.errors import ConfigError
from tayr.eval.false_alarms import false_alarms_per_hour


class TestRate:
    def test_hand_computed_rate(self) -> None:
        """3600 frames at 30fps is exactly 120 seconds. 6 alarms -> 180/hour."""
        frames = [np.array([]) for _ in range(3600)]
        frames[0] = np.array([0.9])
        for i in range(1, 6):
            frames[i * 100] = np.array([0.8])
        rate = false_alarms_per_hour(frames, fps=30.0, confidence_threshold=0.5)
        assert rate.n_false_alarms == 6
        assert rate.duration_hours == pytest.approx(120.0 / 3600.0)
        assert rate.per_hour == pytest.approx(180.0)

    def test_threshold_is_applied(self) -> None:
        frames = [np.array([0.9, 0.4, 0.2])]
        assert false_alarms_per_hour(frames, fps=30, confidence_threshold=0.5).n_false_alarms == 1
        assert false_alarms_per_hour(frames, fps=30, confidence_threshold=0.1).n_false_alarms == 3

    def test_threshold_is_inclusive(self) -> None:
        frames = [np.array([0.5])]
        assert false_alarms_per_hour(frames, fps=30, confidence_threshold=0.5).n_false_alarms == 1

    def test_multiple_alarms_in_one_frame_all_count(self) -> None:
        frames = [np.array([0.9, 0.8, 0.7]), np.array([])]
        assert false_alarms_per_hour(frames, fps=30, confidence_threshold=0.5).n_false_alarms == 3

    def test_clean_footage_gives_zero(self) -> None:
        rate = false_alarms_per_hour(
            [np.array([]) for _ in range(100)], fps=25, confidence_threshold=0.5
        )
        assert rate.n_false_alarms == 0
        assert rate.per_hour == 0.0

    def test_empty_frames_are_the_denominator(self) -> None:
        """Omitting detection-free frames would inflate the per-frame rate 10x here."""
        sparse = [np.array([0.9])] + [np.array([]) for _ in range(9)]
        rate = false_alarms_per_hour(sparse, fps=10, confidence_threshold=0.5)
        assert rate.n_frames == 10
        assert rate.per_frame == pytest.approx(0.1)

    def test_render_mentions_both_rates(self) -> None:
        out = false_alarms_per_hour([np.array([0.9])], fps=30, confidence_threshold=0.5).render()
        assert "/ hour" in out and "/ frame" in out


class TestRejects:
    @pytest.mark.parametrize("fps", [0.0, -1.0])
    def test_invalid_fps(self, fps: float) -> None:
        with pytest.raises(ConfigError, match="fps must be positive"):
            false_alarms_per_hour([np.array([])], fps=fps, confidence_threshold=0.5)

    def test_no_frames(self) -> None:
        with pytest.raises(ConfigError, match="not a rate"):
            false_alarms_per_hour([], fps=30, confidence_threshold=0.5)

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ConfigError, match="confidence_threshold"):
            false_alarms_per_hour([np.array([])], fps=30, confidence_threshold=1.5)
