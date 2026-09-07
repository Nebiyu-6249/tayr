"""Canonical annotation schema.

Every dataset converges here, and nothing downstream sees a native format.

Two decisions in this schema exist to protect the research question rather than to
model the data conveniently:

**Frames with no objects are kept, never dropped.** An absent target is evidence. Drop
those frames and the false-alarms-per-hour denominator silently shrinks, which inflates
the headline safety metric of the whole project.

**`source_video` is mandatory and `track_id` is explicit about its provenance.** The
unit of analysis for the motion hypothesis is a *track*, not a frame. Evaluation splits
are grouped by video, and a track identity that was inferred rather than annotated is a
weaker fact that the census must be able to report separately.

Boxes are `xyxy`, absolute pixels, matching `tayr.geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from tayr.geometry import FloatArray, validate_xyxy


class ObjectClass(StrEnum):
    """The classes the research question is posed over.

    Most anti-UAV datasets label only DRONE. BIRD and AIRCRAFT are what the
    drone/bird/aircraft confusion matrix needs and what most sources do not provide -
    see docs/RESEARCH.md 5.2. UNKNOWN exists so a converter can carry a labelled object
    whose class does not map, rather than guessing or discarding it.
    """

    DRONE = "drone"
    BIRD = "bird"
    AIRCRAFT = "aircraft"
    UNKNOWN = "unknown"


class TrackIdSource(StrEnum):
    """Where a track identity came from. The census reports these separately."""

    ANNOTATED = "annotated"
    """The dataset supplied explicit per-object track ids."""

    SINGLE_TARGET = "single_target"
    """Inferred: the dataset is single-target, so one video is one track.

    Defensible for a single-object tracking benchmark, and a real assumption that must
    be visible in any count derived from it.
    """

    NONE = "none"
    """No track identity available. These objects cannot contribute to track-level
    training or to the motion arm of the hypothesis test."""


@dataclass(frozen=True, slots=True)
class BoxAnnotation:
    """One annotated object in one frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: ObjectClass
    track_id: int | None = None

    def __post_init__(self) -> None:
        # Validate through the same path everything else uses, so a converter bug
        # surfaces here rather than as a mysteriously bad training curve.
        validate_xyxy([self.x1, self.y1, self.x2, self.y2])

    def as_array(self) -> FloatArray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    """One frame. `objects` may be empty, and an empty frame is meaningful."""

    frame_index: int
    objects: tuple[BoxAnnotation, ...] = ()
    file_name: str | None = None
    """The image file this frame's boxes belong to, relative to the split directory.

    Video datasets leave this None: their frames are addressed by (video, index) and
    `to_coco` synthesises a name from those. Image datasets - Pascal VOC among them -
    have a real filename that a detector must be able to open, and synthesising one
    instead would produce a COCO file whose every image path is wrong.
    """

    @property
    def is_empty(self) -> bool:
        """True when no target is present. Distinct from 'frame not annotated'."""
        return len(self.objects) == 0


@dataclass(frozen=True, slots=True)
class VideoAnnotation:
    """One source video, with its frames in temporal order."""

    source_video: str
    frames: tuple[FrameAnnotation, ...]
    track_id_source: TrackIdSource
    width: int | None = None
    height: int | None = None
    frame_index_base: int = 0
    """0 or 1, as the native format numbered its frames. Preserved for round-tripping;
    downstream code should use position in `frames`, not this value."""

    def __post_init__(self) -> None:
        indices = [f.frame_index for f in self.frames]
        if indices != sorted(indices):
            raise ValueError(
                f"{self.source_video}: frames must be in ascending frame_index order; "
                "track features depend on temporal ordering."
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"{self.source_video}: duplicate frame_index values")

    @property
    def n_boxes(self) -> int:
        return sum(len(f.objects) for f in self.frames)

    @property
    def n_empty_frames(self) -> int:
        return sum(1 for f in self.frames if f.is_empty)

    def boxes_array(self) -> FloatArray:
        """All boxes as an (N, 4) xyxy array. Empty (0, 4) if there are none."""
        rows = [o.as_array() for f in self.frames for o in f.objects]
        if not rows:
            return np.empty((0, 4), dtype=np.float64)
        return np.stack(rows)

    def labels(self) -> npt.NDArray[np.str_]:
        return np.array([o.label.value for f in self.frames for o in f.objects], dtype=np.str_)

    def track_keys(self) -> list[tuple[str, int]]:
        """(source_video, track_id) for every object that has a track identity."""
        return [
            (self.source_video, o.track_id)
            for f in self.frames
            for o in f.objects
            if o.track_id is not None
        ]


@dataclass(frozen=True, slots=True)
class DatasetAnnotation:
    """A whole dataset split in canonical form.

    `redistributable` travels with the data so that any code writing it out can refuse
    to place it somewhere public. It defaults to False at the config layer for the same
    reason: forgetting the flag must not be able to cause a licence breach.
    """

    name: str
    split: str
    videos: tuple[VideoAnnotation, ...]
    licence: str = "UNKNOWN"
    redistributable: bool = False
    notes: tuple[str, ...] = ()
    """Facts a loader established about the native files that the canonical form cannot
    carry - which index base the coordinates were read under, how many annotations
    referenced a missing image, how many objects were flagged difficult.

    They travel with the data because the census and the converter both need them and
    neither can re-derive them: by the time annotations are canonical, the native
    evidence is gone. `take_census` surfaces them next to its own warnings.
    """

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def n_frames(self) -> int:
        return sum(len(v.frames) for v in self.videos)

    @property
    def n_boxes(self) -> int:
        return sum(v.n_boxes for v in self.videos)
