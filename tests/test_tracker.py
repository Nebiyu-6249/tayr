"""Tracker tests.

The behaviour that matters most for Tayr is the low-confidence association pass: a
target fading below threshold must keep its identity, because losing it there breaks
tracks exactly in the small-target regime the research question is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from tayr.config import TrackerConfig
from tayr.errors import GeometryError
from tayr.tracking.tracker import ByteTracker, TrackState


def box_at(cx: float, cy: float, side: float = 20.0) -> list[float]:
    h = side / 2
    return [cx - h, cy - h, cx + h, cy + h]


def run(tracker: ByteTracker, frames: list[tuple[list[list[float]], list[float]]]) -> None:
    for boxes, scores in frames:
        arr = np.array(boxes, dtype=np.float64) if boxes else np.empty((0, 4))
        tracker.update(arr, np.array(scores, dtype=np.float64))


class TestBasicTracking:
    def test_single_target_keeps_one_id(self) -> None:
        t = ByteTracker(TrackerConfig(min_hits=2))
        run(t, [([box_at(10 * i, 100)], [0.9]) for i in range(20)])
        assert len(t.finalise()) == 1
        assert t.finalise()[0].track_id == 1

    def test_two_targets_get_distinct_ids(self) -> None:
        t = ByteTracker(TrackerConfig(min_hits=2))
        run(t, [([box_at(10 * i, 100), box_at(10 * i, 500)], [0.9, 0.9]) for i in range(20)])
        assert len({tr.track_id for tr in t.finalise()}) == 2

    def test_tentative_tracks_are_not_reported(self) -> None:
        """One spurious detection must not surface as a track."""
        t = ByteTracker(TrackerConfig(min_hits=3))
        confirmed = t.update(np.array([box_at(0, 0)]), np.array([0.9]))
        assert confirmed == []
        assert t.tracks[0].state is TrackState.TENTATIVE

    def test_min_hits_promotes_a_track(self) -> None:
        t = ByteTracker(TrackerConfig(min_hits=3))
        for i in range(3):
            confirmed = t.update(np.array([box_at(i, 0)]), np.array([0.9]))
        assert len(confirmed) == 1

    def test_empty_frames_age_tracks_toward_death(self) -> None:
        """Frames with no detections must still be fed in, or dead tracks live forever."""
        t = ByteTracker(TrackerConfig(min_hits=2, max_age=5))
        run(t, [([box_at(i, 0)], [0.9]) for i in range(5)])
        assert len(t.confirmed_tracks) == 1
        for _ in range(10):
            t.update(np.empty((0, 4)), np.array([]))
        assert t.finalise() == []

    def test_track_survives_a_short_gap(self) -> None:
        cfg = TrackerConfig(min_hits=2, max_age=10)
        t = ByteTracker(cfg)
        run(t, [([box_at(5 * i, 100)], [0.9]) for i in range(6)])
        for _ in range(4):
            t.update(np.empty((0, 4)), np.array([]))
        t.update(np.array([box_at(50, 100)]), np.array([0.9]))
        alive = t.finalise()
        assert len(alive) == 1
        assert alive[0].track_id == 1

    def test_mismatched_lengths_raise(self) -> None:
        t = ByteTracker()
        with pytest.raises(GeometryError, match="score"):
            t.update(np.array([box_at(0, 0)]), np.array([0.9, 0.8]))


class TestLowConfidenceAssociation:
    """The reason ByteTrack was chosen over a threshold-and-discard tracker."""

    def test_fading_target_keeps_its_identity(self) -> None:
        """Scores decay below high_threshold mid-track. A tracker that discarded them
        would end with two tracks; ByteTrack's second pass must keep one."""
        cfg = TrackerConfig(high_threshold=0.5, low_threshold=0.1, min_hits=2, max_age=30)
        t = ByteTracker(cfg)
        frames = [([box_at(5 * i, 100)], [0.9]) for i in range(6)]
        frames += [([box_at(5 * i, 100)], [0.2]) for i in range(6, 20)]
        frames += [([box_at(5 * i, 100)], [0.9]) for i in range(20, 26)]
        run(t, frames)
        final = t.finalise()
        assert len(final) == 1, f"identity lost: {len(final)} tracks"
        assert final[0].track_id == 1
        assert final[0].n_observations == 26

    def test_low_confidence_detections_cannot_start_a_track(self) -> None:
        """Otherwise background noise spawns tracks continuously."""
        cfg = TrackerConfig(high_threshold=0.5, low_threshold=0.1, min_hits=2)
        t = ByteTracker(cfg)
        run(t, [([box_at(5 * i, 100)], [0.2]) for i in range(20)])
        assert t.finalise() == []

    def test_below_low_threshold_is_ignored_entirely(self) -> None:
        cfg = TrackerConfig(high_threshold=0.5, low_threshold=0.3, min_hits=2, max_age=3)
        t = ByteTracker(cfg)
        run(t, [([box_at(5 * i, 100)], [0.9]) for i in range(5)])
        run(t, [([box_at(5 * i, 100)], [0.05]) for i in range(5, 20)])
        assert t.finalise() == []


class TestIdentityUnderCrossing:
    def test_crossing_targets_do_not_swap_ids(self) -> None:
        """Two targets converging and separating. Greedy IoU matching switches ids here;
        optimal assignment should not. An id switch mid-track corrupts every motion
        feature computed from it."""
        t = ByteTracker(TrackerConfig(min_hits=2, max_age=10))
        frames = []
        for i in range(40):
            a_y = 100 + i * 5
            b_y = 300 - i * 5
            frames.append(([box_at(200, a_y), box_at(200, b_y)], [0.9, 0.9]))
        run(t, frames)
        final = t.finalise()
        assert len(final) == 2
        # The track that started lower must still end lower - no identity swap.
        by_start = sorted(final, key=lambda tr: tr.observed_boxes[0][1])
        assert by_start[0].observed_boxes[-1][1] > by_start[1].observed_boxes[-1][1]


class TestObservedHistory:
    def test_only_real_detections_are_recorded(self) -> None:
        """Gaps must not be filled with filter predictions: smoothness is one of the
        things the motion features measure, and inventing it would fabricate signal."""
        t = ByteTracker(TrackerConfig(min_hits=2, max_age=10))
        run(t, [([box_at(5 * i, 100)], [0.9]) for i in range(5)])
        for _ in range(3):
            t.update(np.empty((0, 4)), np.array([]))
        t.update(np.array([box_at(40, 100)]), np.array([0.9]))
        track = t.finalise()[0]
        assert track.n_observations == 6
        assert track.observed_frames == [0, 1, 2, 3, 4, 8]

    def test_scores_are_retained(self) -> None:
        t = ByteTracker(TrackerConfig(min_hits=1))
        run(t, [([box_at(i, 0)], [0.9 - i * 0.05]) for i in range(5)])
        assert t.finalise()[0].observed_scores[0] == pytest.approx(0.9)


class TestSmallFastTargetAssociation:
    """IoU association has a hard displacement ceiling of roughly 54% of a box side at
    IoU>=0.3, independent of absolute size. For a 10px target that is 5.4px per frame,
    which is less than a bird flapping at 5Hz actually moves - so pure IoU association
    fails on exactly the targets this project's hypothesis is about.
    """

    @staticmethod
    def _oscillating_frames(
        n: int = 120, side: float = 11.0, hz: float = 5.0, amplitude: float = 8.0, fps: float = 30.0
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        frames = []
        for i in range(n):
            y = 600.0 + amplitude * np.sin(2 * np.pi * hz * i / fps)
            frames.append((np.array([box_at(80 + 3.0 * i, y, side)]), np.array([0.9])))
        return frames

    def test_iou_displacement_ceiling_is_a_fixed_fraction_of_box_side(self) -> None:
        """Pins the arithmetic behind centre_distance_factor's existence. The ceiling is
        scale-invariant, so shrinking the target shrinks the tolerable motion with it."""
        from tayr.geometry import iou as iou_fn

        for side in (10.0, 20.0, 50.0):
            lo, hi = 0.0, side
            for _ in range(50):
                mid = (lo + hi) / 2
                overlap = float(
                    iou_fn(np.array([box_at(0, 0, side)]), np.array([box_at(mid, 0, side)]))[0, 0]
                )
                lo, hi = (mid, hi) if overlap > 0.3 else (lo, mid)
            assert lo / side == pytest.approx(0.538, abs=0.01)

    def test_pure_iou_association_loses_a_small_oscillating_target(self) -> None:
        """Documents the failure this fallback exists to fix. If this ever starts
        passing, the fallback may no longer be needed - check before deleting it."""
        cfg = TrackerConfig(min_hits=3, max_age=30, iou_threshold=0.3, centre_distance_factor=0.0)
        t = ByteTracker(cfg)
        for boxes, scores in self._oscillating_frames():
            t.update(boxes, scores)
        assert t.finalise() == []

    def test_distance_fallback_recovers_the_track(self) -> None:
        cfg = TrackerConfig(min_hits=3, max_age=30, iou_threshold=0.3, centre_distance_factor=2.0)
        t = ByteTracker(cfg)
        frames = self._oscillating_frames()
        for boxes, scores in frames:
            t.update(boxes, scores)
        final = t.finalise()
        assert len(final) == 1
        assert final[0].n_observations == len(frames)

    def test_fallback_does_not_let_a_track_teleport(self) -> None:
        """The fallback must widen association, not abolish it. A detection far from the
        prediction is still a new object, not the old one."""
        cfg = TrackerConfig(min_hits=2, max_age=2, iou_threshold=0.3, centre_distance_factor=2.0)
        t = ByteTracker(cfg)
        run(t, [([box_at(100, 100, 20)], [0.9]) for _ in range(4)])
        first_id = t.confirmed_tracks[0].track_id
        # 2000px away: hundreds of box widths, far outside the gate.
        run(t, [([box_at(2100, 2100, 20)], [0.9]) for _ in range(4)])
        ids = {tr.track_id for tr in t.confirmed_tracks}
        assert first_id not in ids or len(t.tracks) > 1

    def test_iou_match_outranks_a_distance_only_match(self) -> None:
        """Two candidate detections: one overlapping, one merely nearby. The overlapping
        one must win, or the fallback would degrade normal association."""
        cfg = TrackerConfig(min_hits=1, max_age=5, iou_threshold=0.3, centre_distance_factor=3.0)
        t = ByteTracker(cfg)
        t.update(np.array([box_at(100, 100, 20)]), np.array([0.9]))
        t.update(np.array([box_at(102, 100, 20), box_at(140, 100, 20)]), np.array([0.9, 0.95]))
        track = t.confirmed_tracks[0]
        assert track.observed_boxes[-1][0] == pytest.approx(92.0, abs=1.0)
