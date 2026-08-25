"""Geometry tests.

Coordinate transforms and format conversions are where detection-pipeline bugs hide,
because a wrong answer still looks like a plausible bounding box. These tests check
round-trips, boundary conditions, and the specific off-by-offset error that sliced
inference invites.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tayr.errors import GeometryError
from tayr.geometry import (
    SizeBucket,
    area,
    cxcywh_to_xyxy,
    denormalise,
    frame_to_tile,
    iou,
    normalise,
    pixels_on_target,
    size_bucket,
    tile_to_frame,
    validate_xyxy,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_xywh,
)

# Positive-area boxes, kept well away from float64 precision limits.
_coord = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_size = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def valid_xyxy(draw: st.DrawFn) -> np.ndarray:
    x1 = draw(_coord)
    y1 = draw(_coord)
    return np.array([x1, y1, x1 + draw(_size), y1 + draw(_size)], dtype=np.float64)


class TestRoundTrips:
    """Every format conversion must be its own inverse's inverse."""

    @given(valid_xyxy())
    def test_xyxy_xywh_round_trip(self, box: np.ndarray) -> None:
        assert np.allclose(xywh_to_xyxy(xyxy_to_xywh(box)), box, rtol=1e-9, atol=1e-9)

    @given(valid_xyxy())
    def test_xyxy_cxcywh_round_trip(self, box: np.ndarray) -> None:
        assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(box)), box, rtol=1e-9, atol=1e-9)

    @given(valid_xyxy(), st.integers(1, 8000), st.integers(1, 8000))
    def test_normalise_round_trip(self, box: np.ndarray, w: int, h: int) -> None:
        assert np.allclose(denormalise(normalise(box, width=w, height=h), width=w, height=h), box)

    @given(valid_xyxy(), st.integers(-4000, 4000), st.integers(-4000, 4000))
    def test_tile_frame_round_trip(self, box: np.ndarray, ox: int, oy: int) -> None:
        assert np.allclose(
            frame_to_tile(tile_to_frame(box, offset_x=ox, offset_y=oy), offset_x=ox, offset_y=oy),
            box,
        )


class TestKnownValues:
    """Hand-computed cases. Round-trips alone would pass even if both directions
    shared the same sign error, so the absolute values are pinned too."""

    def test_xyxy_to_xywh_known(self) -> None:
        assert np.allclose(xyxy_to_xywh([10.0, 20.0, 30.0, 60.0]), [10.0, 20.0, 20.0, 40.0])

    def test_xyxy_to_cxcywh_known(self) -> None:
        assert np.allclose(xyxy_to_cxcywh([10.0, 20.0, 30.0, 60.0]), [20.0, 40.0, 20.0, 40.0])

    def test_tile_to_frame_shifts_both_corners(self) -> None:
        # The classic sliced-inference bug is shifting only the top-left corner,
        # which silently stretches every box by the tile offset.
        out = tile_to_frame([5.0, 5.0, 15.0, 25.0], offset_x=640, offset_y=1280)
        assert np.allclose(out, [645.0, 1285.0, 655.0, 1305.0])
        assert np.isclose(area(out), area([5.0, 5.0, 15.0, 25.0]))

    def test_area_and_pixels_on_target(self) -> None:
        assert np.isclose(area([0.0, 0.0, 20.0, 20.0]), 400.0)
        assert np.isclose(pixels_on_target([0.0, 0.0, 20.0, 20.0]), 20.0)
        # sqrt(area) is the documented measure: a 40x10 box is also 20px on target.
        assert np.isclose(pixels_on_target([0.0, 0.0, 40.0, 10.0]), 20.0)


class TestBatching:
    """Transforms are vectorised over leading dimensions, and must not reorder."""

    def test_preserves_shape(self) -> None:
        boxes = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
        boxes[..., 2] = boxes[..., 0] + 5.0
        boxes[..., 3] = boxes[..., 1] + 7.0
        assert xyxy_to_xywh(boxes).shape == (2, 3, 4)
        assert size_bucket(boxes).shape == (2, 3)

    def test_batch_matches_elementwise(self) -> None:
        boxes = np.array([[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 50.0, 50.0]])
        batched = xyxy_to_xywh(boxes)
        for i, single in enumerate(boxes):
            assert np.allclose(batched[i], xyxy_to_xywh(single))


class TestSizeBuckets:
    """Bucket edges are half-open [lo, hi). These are the buckets the research
    question is evaluated in, so an off-by-one at a boundary would misattribute
    results between 'appearance works' and 'appearance has saturated'."""

    @pytest.mark.parametrize(
        ("side", "expected"),
        [
            (4.0, SizeBucket.TINY),
            (7.999, SizeBucket.TINY),
            (8.0, SizeBucket.SMALL),  # boundary: 8 is SMALL, not TINY
            (15.999, SizeBucket.SMALL),
            (16.0, SizeBucket.MEDIUM),  # boundary
            (31.999, SizeBucket.MEDIUM),
            (32.0, SizeBucket.LARGE),  # boundary
            (500.0, SizeBucket.LARGE),
        ],
    )
    def test_boundaries(self, side: float, expected: SizeBucket) -> None:
        assert size_bucket([0.0, 0.0, side, side]).item() == expected.value

    def test_every_box_lands_in_exactly_one_bucket(self) -> None:
        sides = np.linspace(0.5, 200.0, 400)
        boxes = np.stack([np.zeros_like(sides), np.zeros_like(sides), sides, sides], axis=-1)
        labels = size_bucket(boxes)
        assert set(labels.tolist()) <= {b.value for b in SizeBucket}
        assert len(labels) == len(sides)


class TestIoU:
    def test_identical_boxes(self) -> None:
        b = [[0.0, 0.0, 10.0, 10.0]]
        assert np.isclose(iou(b, b)[0, 0], 1.0)

    def test_disjoint_boxes(self) -> None:
        assert np.isclose(iou([[0.0, 0.0, 10.0, 10.0]], [[50.0, 50.0, 60.0, 60.0]])[0, 0], 0.0)

    def test_half_overlap(self) -> None:
        # 10x10 and 10x10 sharing a 5x10 strip: inter 50, union 150.
        got = iou([[0.0, 0.0, 10.0, 10.0]], [[5.0, 0.0, 15.0, 10.0]])[0, 0]
        assert np.isclose(got, 50.0 / 150.0)

    def test_matrix_shape(self) -> None:
        a = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [2.0, 2.0, 12.0, 12.0]])
        b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        assert iou(a, b).shape == (3, 2)

    def test_small_box_iou_sensitivity(self) -> None:
        """Documents, as an executable fact, why mAP@0.5 is near-useless below 8px.

        An 8x8 box displaced by 2px in one axis loses enough overlap that a 0.5
        threshold is already marginal. This is the numeric justification for the
        eval spec reporting mAP@0.25 in the small buckets.
        """
        gt = [[0.0, 0.0, 8.0, 8.0]]
        shifted = [[2.0, 0.0, 10.0, 8.0]]
        got = iou(gt, shifted)[0, 0]
        assert np.isclose(got, 48.0 / 80.0)  # 0.6 - one 2px slip and it is borderline
        worse = iou(gt, [[3.0, 0.0, 11.0, 8.0]])[0, 0]
        assert worse < 0.5  # 3px and a correct detection is scored as a miss


class TestFailsLoudly:
    """Invalid input raises. It is never clamped, repaired, or silently dropped."""

    def test_wrong_trailing_dimension(self) -> None:
        with pytest.raises(GeometryError, match=r"shape \(\.\.\., 4\)"):
            xyxy_to_xywh([1.0, 2.0, 3.0])

    def test_nan_rejected(self) -> None:
        with pytest.raises(GeometryError, match="NaN or inf"):
            xyxy_to_xywh([0.0, 0.0, np.nan, 5.0])

    def test_negative_width_rejected(self) -> None:
        with pytest.raises(GeometryError, match="non-positive width or height"):
            validate_xyxy([10.0, 0.0, 5.0, 20.0])

    def test_zero_area_rejected(self) -> None:
        with pytest.raises(GeometryError, match="non-positive width or height"):
            validate_xyxy([10.0, 10.0, 10.0, 20.0])

    def test_error_names_the_offending_index(self) -> None:
        boxes = [[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0], [9.0, 0.0, 1.0, 5.0]]
        with pytest.raises(GeometryError, match=r"index \[2\]"):
            validate_xyxy(boxes)

    def test_zero_image_size_rejected(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            normalise([0.0, 0.0, 5.0, 5.0], width=0, height=10)
