"""Detection: the detector interface, and sliced-inference geometry."""

from tayr.detection.slicing import Tile, generate_tiles, merge_tile_detections, nms

__all__ = ["Tile", "generate_tiles", "merge_tile_detections", "nms"]
