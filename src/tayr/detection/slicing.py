"""Sliced (tiled) inference geometry.

A 15px drone in a 3840x2160 frame is roughly 0.003% of the pixels. Resize that frame to
a 640x640 detector input and the target is under 3px - gone. Slicing runs the detector
on native-resolution crops instead, which is what makes small targets detectable at all.

The cost is real and worth stating: tile count scales as roughly
`(W/tw) * (H/th) / (1-overlap)^2`. At 4K with 640px tiles and 0.2 overlap that is an 8x4
grid - 32 forward passes per frame, so 30fps video needs ~960 passes per second. That
figure is pinned by a test, and it is exactly why the region-proposal stage exists -
see docs/RESEARCH.md 3.2.

Two bugs live here and both are tested directly:

  1. **Coordinate drift.** A detection found in tile-local coordinates must be shifted
     by the tile origin. Shifting only one corner stretches every box.
  2. **Tile-boundary duplicates.** Overlap means one object can be detected in two
     adjacent tiles, producing two boxes for one drone. The merge step must dedupe, or
     every count downstream is inflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tayr.errors import ConfigError, GeometryError
from tayr.geometry import FloatArray, iou, tile_to_frame


@dataclass(frozen=True, slots=True)
class Tile:
    """A crop region in frame coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


def generate_tiles(
    *, frame_width: int, frame_height: int, tile_width: int, tile_height: int, overlap: float
) -> list[Tile]:
    """Cover a frame with overlapping tiles.

    Tiles are clamped to the frame edge rather than allowed to run past it, so the
    final row and column stay full-size and shift inward. A partial edge tile would
    feed the detector a differently-scaled input and produce systematically worse
    detections along two edges of every frame.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ConfigError(f"frame size must be positive; got {frame_width}x{frame_height}")
    if tile_width <= 0 or tile_height <= 0:
        raise ConfigError(f"tile size must be positive; got {tile_width}x{tile_height}")
    if not 0.0 <= overlap < 1.0:
        raise ConfigError(f"overlap must be in [0, 1); got {overlap}")

    # A tile larger than the frame degenerates to a single full-frame tile.
    tw = min(tile_width, frame_width)
    th = min(tile_height, frame_height)

    step_x = max(1, round(tw * (1.0 - overlap)))
    step_y = max(1, round(th * (1.0 - overlap)))

    xs = list(range(0, max(1, frame_width - tw + 1), step_x))
    ys = list(range(0, max(1, frame_height - th + 1), step_y))
    if xs[-1] != frame_width - tw:
        xs.append(frame_width - tw)
    if ys[-1] != frame_height - th:
        ys.append(frame_height - th)

    return [Tile(x=x, y=y, width=tw, height=th) for y in ys for x in xs]


def nms(
    boxes: FloatArray, scores: npt.NDArray[np.float64], *, iou_threshold: float
) -> npt.NDArray[np.int64]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    if not 0.0 < iou_threshold <= 1.0:
        raise GeometryError(f"iou_threshold must be in (0, 1]; got {iou_threshold}")

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise GeometryError(f"{len(boxes)} box(es) but {len(scores)} score(s)")
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    order: list[int] = [int(i) for i in np.argsort(-scores, kind="stable")]
    keep: list[int] = []

    while order:
        current = int(order.pop(0))
        keep.append(current)
        if not order:
            break
        remaining = np.array(order, dtype=np.int64)
        overlaps = iou(boxes[current][None, :], boxes[remaining])[0]
        order = [int(i) for i, ov in zip(remaining, overlaps, strict=True) if ov < iou_threshold]

    return np.array(keep, dtype=np.int64)


def merge_tile_detections(
    per_tile_boxes: list[FloatArray],
    per_tile_scores: list[npt.NDArray[np.float64]],
    tiles: list[Tile],
    *,
    iou_threshold: float = 0.5,
) -> tuple[FloatArray, npt.NDArray[np.float64]]:
    """Map per-tile detections into frame coordinates and suppress duplicates.

    Overlapping tiles mean one object can be reported twice. Without this step, a drone
    detected in two adjacent tiles counts as two drones, and every downstream count -
    detections, tracks, false alarms - is inflated.
    """
    if not (len(per_tile_boxes) == len(per_tile_scores) == len(tiles)):
        raise ConfigError(
            f"mismatched lengths: {len(per_tile_boxes)} box array(s), "
            f"{len(per_tile_scores)} score array(s), {len(tiles)} tile(s)"
        )

    all_boxes: list[FloatArray] = []
    all_scores: list[npt.NDArray[np.float64]] = []

    for boxes, scores, tile in zip(per_tile_boxes, per_tile_scores, tiles, strict=True):
        arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        sc = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(arr) != len(sc):
            raise ConfigError(
                f"tile at ({tile.x},{tile.y}): {len(arr)} box(es), {len(sc)} score(s)"
            )
        if len(arr) == 0:
            continue
        all_boxes.append(tile_to_frame(arr, offset_x=tile.x, offset_y=tile.y))
        all_scores.append(sc)

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64)

    boxes_cat = np.concatenate(all_boxes, axis=0)
    scores_cat = np.concatenate(all_scores, axis=0)
    kept = nms(boxes_cat, scores_cat, iou_threshold=iou_threshold)
    return boxes_cat[kept], scores_cat[kept]
