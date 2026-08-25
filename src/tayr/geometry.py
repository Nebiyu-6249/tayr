"""Bounding-box formats, coordinate transforms, and pixels-on-target bucketing.

Three box formats are used across the datasets Tayr ingests, and mixing them up is the
single most common source of silent bugs in a detection pipeline:

    xyxy    (x1, y1, x2, y2)  absolute corners.       Used internally everywhere.
    xywh    (x, y, w, h)      top-left + size.        COCO, Drone-vs-Bird.
    cxcywh  (cx, cy, w, h)    centre + size.          YOLO (usually normalised).

`xyxy` is Tayr's internal canonical format. Converters at the dataset boundary turn
everything into `xyxy` immediately, and nothing downstream handles any other format.

All functions take and return float arrays of shape (..., 4) and are vectorised over
leading dimensions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from tayr.errors import GeometryError

FloatArray = npt.NDArray[np.float64]

# Pixel-on-target buckets from the evaluation spec. Stored as half-open intervals
# [lo, hi) in units of sqrt(area), so every box lands in exactly one bucket and the
# boundaries are unambiguous. A box of exactly 32px is LARGE, not MEDIUM.
_BUCKET_EDGES: Final = (8.0, 16.0, 32.0)


class SizeBucket(StrEnum):
    """Pixels-on-target bucket. The research question lives in TINY and SMALL."""

    TINY = "<8px"
    SMALL = "8-16px"
    MEDIUM = "16-32px"
    LARGE = ">32px"


def _as_boxes(boxes: npt.ArrayLike, *, name: str = "boxes") -> FloatArray:
    """Coerce to a float64 array of shape (..., 4), raising on anything else."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 4:
        raise GeometryError(
            f"{name} must have shape (..., 4); got {arr.shape}. "
            "Boxes are (x1,y1,x2,y2), (x,y,w,h) or (cx,cy,w,h) depending on the format."
        )
    if not np.all(np.isfinite(arr)):
        raise GeometryError(f"{name} contains NaN or inf; refusing to guess a repair.")
    return arr


def validate_xyxy(boxes: npt.ArrayLike) -> FloatArray:
    """Return `boxes` as xyxy, raising if any box has non-positive width or height.

    Degenerate boxes are a converter bug. Clamping them here would hide that bug and
    quietly corrupt every metric computed downstream.
    """
    arr = _as_boxes(boxes)
    widths = arr[..., 2] - arr[..., 0]
    heights = arr[..., 3] - arr[..., 1]
    bad = (widths <= 0) | (heights <= 0)
    if np.any(bad):
        idx = np.argwhere(bad)
        raise GeometryError(
            f"{int(bad.sum())} box(es) have non-positive width or height "
            f"(first at index {idx[0].tolist()}: {arr[tuple(idx[0])].tolist()}). "
            "x2 must exceed x1 and y2 must exceed y1."
        )
    return arr


def xywh_to_xyxy(boxes: npt.ArrayLike) -> FloatArray:
    """(x, y, w, h) top-left+size -> (x1, y1, x2, y2)."""
    arr = _as_boxes(boxes)
    x, y, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


def xyxy_to_xywh(boxes: npt.ArrayLike) -> FloatArray:
    """(x1, y1, x2, y2) -> (x, y, w, h) top-left+size."""
    arr = _as_boxes(boxes)
    x1, y1, x2, y2 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)


def cxcywh_to_xyxy(boxes: npt.ArrayLike) -> FloatArray:
    """(cx, cy, w, h) centre+size -> (x1, y1, x2, y2)."""
    arr = _as_boxes(boxes)
    cx, cy, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def xyxy_to_cxcywh(boxes: npt.ArrayLike) -> FloatArray:
    """(x1, y1, x2, y2) -> (cx, cy, w, h) centre+size."""
    arr = _as_boxes(boxes)
    x1, y1, x2, y2 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def normalise(boxes: npt.ArrayLike, *, width: int, height: int) -> FloatArray:
    """Scale absolute pixel coordinates into [0, 1] against an image size.

    Applies to any format whose channels are (x-like, y-like, x-like, y-like), which
    covers all three formats Tayr uses.
    """
    if width <= 0 or height <= 0:
        raise GeometryError(f"image size must be positive; got {width}x{height}")
    arr = _as_boxes(boxes)
    scale = np.array([width, height, width, height], dtype=np.float64)
    return arr / scale


def denormalise(boxes: npt.ArrayLike, *, width: int, height: int) -> FloatArray:
    """Inverse of `normalise`."""
    if width <= 0 or height <= 0:
        raise GeometryError(f"image size must be positive; got {width}x{height}")
    arr = _as_boxes(boxes)
    scale = np.array([width, height, width, height], dtype=np.float64)
    return arr * scale


def tile_to_frame(boxes: npt.ArrayLike, *, offset_x: int, offset_y: int) -> FloatArray:
    """Map xyxy boxes from tile-local coordinates back to whole-frame coordinates.

    Sliced inference detects in a crop and must report in frame coordinates. Getting
    this wrong shifts every detection by the tile origin, which looks like a badly
    trained detector rather than an off-by-offset bug - so it is tested directly.
    """
    arr = _as_boxes(boxes)
    shift = np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float64)
    return arr + shift


def frame_to_tile(boxes: npt.ArrayLike, *, offset_x: int, offset_y: int) -> FloatArray:
    """Inverse of `tile_to_frame`."""
    arr = _as_boxes(boxes)
    shift = np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float64)
    return arr - shift


def area(boxes: npt.ArrayLike) -> FloatArray:
    """Box area in square pixels, from xyxy."""
    arr = validate_xyxy(boxes)
    return (arr[..., 2] - arr[..., 0]) * (arr[..., 3] - arr[..., 1])


def pixels_on_target(boxes: npt.ArrayLike) -> FloatArray:
    """Pixels-on-target as sqrt(area), from xyxy.

    sqrt(area) is the geometric mean side length. It is the measure used to define the
    evaluation size buckets, and it is reported alongside every bucketed metric so the
    definition is never ambiguous in the writeup. A 20x20 box is 20px on target; so is
    a 40x10 box.
    """
    return np.sqrt(area(boxes))


def size_bucket(boxes: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Assign each xyxy box to a `SizeBucket` by pixels-on-target.

    Buckets are half-open [lo, hi): a box of exactly 8.0px is SMALL, exactly 16.0px is
    MEDIUM, exactly 32.0px is LARGE.
    """
    pot = pixels_on_target(boxes)
    idx = np.digitize(pot, _BUCKET_EDGES, right=False)
    labels = np.array(
        [SizeBucket.TINY, SizeBucket.SMALL, SizeBucket.MEDIUM, SizeBucket.LARGE],
        dtype=np.str_,
    )
    return labels[idx]


def iou(boxes_a: npt.ArrayLike, boxes_b: npt.ArrayLike) -> FloatArray:
    """Pairwise IoU matrix of shape (N, M) between two sets of xyxy boxes.

    Note for the evaluation harness: IoU is brutally sensitive at small sizes. On an
    8x8 box a 2px offset in one axis already costs roughly a third of the overlap, so
    mAP@0.5 in the TINY bucket sits close to the annotation noise floor. This is why
    the eval spec also reports mAP@0.25 below 16px - see docs/RESEARCH.md 6.1.
    """
    a = validate_xyxy(boxes_a).reshape(-1, 4)
    b = validate_xyxy(boxes_b).reshape(-1, 4)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    result: FloatArray = inter / union
    return result
