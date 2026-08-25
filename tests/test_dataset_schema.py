"""Schema invariant tests.

These guards exist because track feature extraction depends on temporal ordering, and
an out-of-order or duplicated frame would produce a velocity profile that is wrong in a
way nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from tayr.datasets.schema import (
    BoxAnnotation,
    DatasetAnnotation,
    FrameAnnotation,
    ObjectClass,
    TrackIdSource,
    VideoAnnotation,
)
from tayr.errors import GeometryError


def _frame(i: int, *boxes: BoxAnnotation) -> FrameAnnotation:
    return FrameAnnotation(frame_index=i, objects=boxes)


class TestBoxAnnotation:
    def test_degenerate_box_rejected_at_construction(self) -> None:
        """Validation happens in the schema, so a converter bug surfaces at the point
        of the bug rather than as a bad training curve."""
        with pytest.raises(GeometryError, match="non-positive width or height"):
            BoxAnnotation(10.0, 10.0, 5.0, 20.0, ObjectClass.DRONE)

    def test_as_array_is_xyxy(self) -> None:
        b = BoxAnnotation(1.0, 2.0, 3.0, 4.0, ObjectClass.DRONE)
        assert np.array_equal(b.as_array(), np.array([1.0, 2.0, 3.0, 4.0]))


class TestVideoAnnotation:
    def test_rejects_out_of_order_frames(self) -> None:
        with pytest.raises(ValueError, match="ascending frame_index order"):
            VideoAnnotation("v", (_frame(2), _frame(1)), TrackIdSource.NONE)

    def test_rejects_duplicate_frame_indices(self) -> None:
        with pytest.raises(ValueError, match="duplicate frame_index"):
            VideoAnnotation("v", (_frame(1), _frame(1)), TrackIdSource.NONE)

    def test_boxes_array_shape_and_empty_case(self) -> None:
        box = BoxAnnotation(0.0, 0.0, 5.0, 5.0, ObjectClass.DRONE, 1)
        v = VideoAnnotation("v", (_frame(0, box), _frame(1)), TrackIdSource.ANNOTATED)
        assert v.boxes_array().shape == (1, 4)
        assert v.n_boxes == 1
        assert v.n_empty_frames == 1

        empty = VideoAnnotation("v", (_frame(0), _frame(1)), TrackIdSource.NONE)
        assert empty.boxes_array().shape == (0, 4)

    def test_labels_and_track_keys(self) -> None:
        v = VideoAnnotation(
            "vid",
            (
                _frame(0, BoxAnnotation(0.0, 0.0, 5.0, 5.0, ObjectClass.DRONE, 7)),
                _frame(1, BoxAnnotation(0.0, 0.0, 5.0, 5.0, ObjectClass.BIRD, None)),
            ),
            TrackIdSource.ANNOTATED,
        )
        assert v.labels().tolist() == ["drone", "bird"]
        # The untracked object contributes no track key.
        assert v.track_keys() == [("vid", 7)]


class TestDatasetAnnotation:
    def test_aggregates_and_defaults_to_non_redistributable(self) -> None:
        box = BoxAnnotation(0.0, 0.0, 5.0, 5.0, ObjectClass.DRONE, 0)
        ds = DatasetAnnotation(
            name="d",
            split="train",
            videos=(
                VideoAnnotation("a", (_frame(0, box), _frame(1)), TrackIdSource.ANNOTATED),
                VideoAnnotation("b", (_frame(0, box),), TrackIdSource.ANNOTATED),
            ),
        )
        assert (ds.n_videos, ds.n_frames, ds.n_boxes) == (2, 3, 2)
        assert ds.redistributable is False
        assert ds.licence == "UNKNOWN"
