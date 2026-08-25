"""ByteTrack-style multi-object tracking.

The core idea, from the ByteTrack paper's reference implementation
`[VERIFIED: https://github.com/FoundationVision/ByteTrack - MIT licensed; the README
states the method works "by associating every detection box instead of only the high
score ones. For the low score detection boxes, we utilize their similarities with
tracklets to recover true objects and filter out the background detections"]`:

Association runs in two passes.

  1. High-confidence detections are matched to all active tracks.
  2. **Low-confidence detections are matched to whatever tracks remain unmatched.**

Pass 2 is the entire point for Tayr. A drone fading to twelve pixels against bright sky
produces a low-score detection, not an absent one. A tracker that discards everything
below its threshold breaks the track exactly when the target gets small - which is
precisely the regime this project's research question is about. Only high-confidence
detections may *start* a new track, so noise does not spawn tracks.

Implemented here rather than taken from BoxMOT, which is AGPL-3.0
`[VERIFIED: https://github.com/mikel-brostrom/boxmot]` and would relicense this project.

Assignment uses `scipy.optimize.linear_sum_assignment` (optimal) rather than greedy
matching by descending IoU. Greedy switches identities exactly when two targets cross,
and an id switch mid-track corrupts every motion feature computed from that track - the
signal the whole hypothesis rests on.

SCOPE: the filter's one-frame prediction exists for association only. See the package
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from tayr.config import TrackerConfig
from tayr.errors import GeometryError
from tayr.geometry import FloatArray, iou, pixels_on_target, xyxy_to_cxcywh
from tayr.tracking.kalman import KalmanBoxFilter


class TrackState(StrEnum):
    TENTATIVE = "tentative"
    """Seen, but not yet `min_hits` times. Not reported: reporting these would let a
    single spurious detection appear as a track."""

    CONFIRMED = "confirmed"
    LOST = "lost"
    """Unmatched recently but still within `max_age`; may yet be recovered."""

    REMOVED = "removed"


@dataclass
class Track:
    """One tracked object and its observed history."""

    track_id: int
    filter: KalmanBoxFilter
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    start_frame: int = 0
    # Observed history, used by feature extraction. Only frames where a detection was
    # actually associated are recorded: filling gaps with the filter's own predictions
    # would manufacture smooth motion that was never measured.
    observed_frames: list[int] = field(default_factory=list)
    observed_boxes: list[FloatArray] = field(default_factory=list)
    observed_scores: list[float] = field(default_factory=list)

    @property
    def box_xyxy(self) -> FloatArray:
        return self.filter.box_xyxy

    @property
    def n_observations(self) -> int:
        return len(self.observed_boxes)

    def record(self, frame_index: int, box: FloatArray, score: float) -> None:
        self.observed_frames.append(frame_index)
        self.observed_boxes.append(np.asarray(box, dtype=np.float64).reshape(4))
        self.observed_scores.append(float(score))


class ByteTracker:
    """Associates detections into tracks across frames."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self._frame_index = -1

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state is TrackState.CONFIRMED]

    def update(
        self,
        boxes_xyxy: FloatArray,
        scores: npt.NDArray[np.float64],
        *,
        frame_index: int | None = None,
    ) -> list[Track]:
        """Consume one frame of detections. Returns the currently confirmed tracks.

        Call once per frame **including frames with no detections** - an empty frame is
        what ages a track toward death, and skipping those frames would keep dead tracks
        alive indefinitely.
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(boxes) != len(scores):
            raise GeometryError(f"{len(boxes)} box(es) but {len(scores)} score(s)")

        self._frame_index = frame_index if frame_index is not None else self._frame_index + 1
        cfg = self.config

        for track in self.tracks:
            track.filter.predict()
            track.age += 1
            track.time_since_update += 1

        high = scores >= cfg.high_threshold
        low = (scores >= cfg.low_threshold) & ~high

        active = [t for t in self.tracks if t.state is not TrackState.REMOVED]

        # --- Pass 1: high-confidence detections against every active track ---
        matched, unmatched_tracks, unmatched_high = self._associate(
            active, boxes[high], scores[high], cfg.iou_threshold
        )
        for track, det_i in matched:
            self._apply(track, boxes[high][det_i], float(scores[high][det_i]))

        # --- Pass 2: low-confidence detections against tracks still unmatched ---
        # This is the step that keeps a fading target's identity alive.
        if len(boxes[low]):
            still_unmatched = [active[i] for i in unmatched_tracks]
            matched_low, unmatched_tracks_2, _ = self._associate(
                still_unmatched, boxes[low], scores[low], cfg.iou_threshold
            )
            for track, det_i in matched_low:
                self._apply(track, boxes[low][det_i], float(scores[low][det_i]))
            leftover = [still_unmatched[i] for i in unmatched_tracks_2]
        else:
            leftover = [active[i] for i in unmatched_tracks]

        for track in leftover:
            if track.state is TrackState.TENTATIVE:
                # An unconfirmed track that misses immediately was probably noise.
                track.state = TrackState.REMOVED
            elif track.time_since_update > cfg.max_age:
                track.state = TrackState.REMOVED
            else:
                track.state = TrackState.LOST

        # Only high-confidence detections may start a track.
        for det_i in unmatched_high:
            self._spawn(boxes[high][det_i], float(scores[high][det_i]))

        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]
        return self.confirmed_tracks

    def finalise(self) -> list[Track]:
        """End the sequence and return every track that was ever confirmed.

        Call after the last frame. Tracks still alive at the end are legitimate results -
        a target that is still in view when the video stops was tracked successfully.
        """
        return [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]

    # ------------------------------------------------------------------ internals

    def _affinity(self, track_boxes: FloatArray, boxes: FloatArray, threshold: float) -> FloatArray:
        """Association affinity in [0, 1]. Zero means the pair is not eligible.

        IoU at or above `threshold` is the primary signal. Below it, a size-normalised
        centre distance provides a fallback, because IoU cannot span the per-frame
        displacement of a small fast target: its ceiling is about 54% of a box side at
        IoU>=0.3 regardless of absolute size, which for a 10px target is 5.4px per frame
        - less than a bird flapping at 5Hz actually moves.

        Fallback affinities are compressed strictly below `threshold`, so a genuine IoU
        match always outranks a distance-only one in the assignment.
        """
        iou_matrix = iou(track_boxes, boxes)
        affinity = np.where(iou_matrix >= threshold, iou_matrix, 0.0)

        factor = self.config.centre_distance_factor
        if factor <= 0.0:
            return affinity

        track_centres = xyxy_to_cxcywh(track_boxes)[:, :2]
        det_centres = xyxy_to_cxcywh(boxes)[:, :2]
        distances = np.linalg.norm(track_centres[:, None, :] - det_centres[None, :, :], axis=-1)

        # Normalise by the track's own size: "within N box widths", not N pixels.
        scale = np.maximum(pixels_on_target(track_boxes), 1e-6)[:, None]
        normalised = distances / (factor * scale)
        fallback = np.clip(1.0 - normalised, 0.0, 1.0) * threshold * 0.999

        result: FloatArray = np.maximum(affinity, fallback)
        return result

    def _associate(
        self,
        tracks: list[Track],
        boxes: FloatArray,
        scores: npt.NDArray[np.float64],
        threshold: float,
    ) -> tuple[list[tuple[Track, int]], list[int], list[int]]:
        """Optimal assignment. Returns (matches, unmatched track idx, unmatched det idx)."""
        del scores  # association is by geometry only; confidence chose the pass
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))

        track_boxes = np.stack([t.box_xyxy for t in tracks])
        affinity = self._affinity(track_boxes, boxes, threshold)

        # Hungarian minimises cost, so maximise affinity by negating.
        track_idx, det_idx = linear_sum_assignment(-affinity)

        matches: list[tuple[Track, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for t_i, d_i in zip(track_idx, det_idx, strict=True):
            # The optimal assignment still pairs things that are not plausibly the
            # same object; reject those or a track would teleport to an unrelated
            # detection.
            if affinity[t_i, d_i] <= 0.0:
                continue
            matches.append((tracks[int(t_i)], int(d_i)))
            matched_tracks.add(int(t_i))
            matched_dets.add(int(d_i))

        return (
            matches,
            [i for i in range(len(tracks)) if i not in matched_tracks],
            [i for i in range(len(boxes)) if i not in matched_dets],
        )

    def _apply(self, track: Track, box: FloatArray, score: float) -> None:
        track.filter.update(box)
        track.hits += 1
        track.time_since_update = 0
        track.record(self._frame_index, box, score)
        # A LOST track is recovered the moment anything matches it again - that
        # recovery is what pass 2 exists to enable.
        recovered = track.state is TrackState.LOST
        promoted = track.state is TrackState.TENTATIVE and track.hits >= self.config.min_hits
        if recovered or promoted:
            track.state = TrackState.CONFIRMED

    def _spawn(self, box: FloatArray, score: float) -> None:
        track = Track(
            track_id=self._next_id,
            filter=KalmanBoxFilter(box),
            start_frame=self._frame_index,
        )
        track.record(self._frame_index, box, score)
        if self.config.min_hits <= 1:
            track.state = TrackState.CONFIRMED
        self.tracks.append(track)
        self._next_id += 1
