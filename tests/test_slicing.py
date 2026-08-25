"""Sliced-inference tests.

Two failure modes get direct attention: coordinate drift when mapping tile-local
detections back to the frame, and duplicate detections across overlapping tile seams.
"""

from __future__ import annotations

import numpy as np
import pytest

from tayr.detection.slicing import Tile, generate_tiles, merge_tile_detections, nms
from tayr.errors import ConfigError, GeometryError


class TestTileGeneration:
    def test_tiles_cover_every_pixel(self) -> None:
        w, h = 1920, 1080
        tiles = generate_tiles(
            frame_width=w, frame_height=h, tile_width=640, tile_height=640, overlap=0.2
        )
        covered = np.zeros((h, w), dtype=bool)
        for t in tiles:
            covered[t.y : t.y2, t.x : t.x2] = True
        assert covered.all(), "tile grid leaves uncovered pixels"

    def test_tiles_stay_inside_the_frame(self) -> None:
        tiles = generate_tiles(
            frame_width=1000, frame_height=700, tile_width=640, tile_height=640, overlap=0.2
        )
        for t in tiles:
            assert t.x >= 0 and t.y >= 0
            assert t.x2 <= 1000 and t.y2 <= 700

    def test_all_tiles_are_full_size(self) -> None:
        """Edge tiles shift inward rather than shrinking: a partial tile would feed the
        detector a differently-scaled input along two edges of every frame."""
        tiles = generate_tiles(
            frame_width=1000, frame_height=700, tile_width=640, tile_height=640, overlap=0.2
        )
        assert all(t.width == 640 and t.height == 640 for t in tiles)

    def test_tile_count_matches_the_documented_cost_model(self) -> None:
        """Pins the cost figure quoted in docs/RESEARCH.md 3.2, so the doc cannot drift
        away from what the code actually does. 4K at 640px tiles with 0.2 overlap is an
        8x4 grid: 32 tiles, hence ~960 detector forward passes per second at 30fps."""
        tiles = generate_tiles(
            frame_width=3840, frame_height=2160, tile_width=640, tile_height=640, overlap=0.2
        )
        assert len(tiles) == 32
        assert len({t.x for t in tiles}) == 8
        assert len({t.y for t in tiles}) == 4

    def test_zero_overlap_gives_a_clean_grid(self) -> None:
        tiles = generate_tiles(
            frame_width=1280, frame_height=640, tile_width=640, tile_height=640, overlap=0.0
        )
        assert len(tiles) == 2
        assert [(t.x, t.y) for t in tiles] == [(0, 0), (640, 0)]

    def test_more_overlap_means_more_tiles(self) -> None:
        def n(ov: float) -> int:
            return len(
                generate_tiles(
                    frame_width=1920,
                    frame_height=1080,
                    tile_width=640,
                    tile_height=640,
                    overlap=ov,
                )
            )

        assert n(0.0) < n(0.2) < n(0.5)

    def test_tile_larger_than_frame_degenerates_to_one_tile(self) -> None:
        tiles = generate_tiles(
            frame_width=320, frame_height=240, tile_width=640, tile_height=640, overlap=0.2
        )
        assert len(tiles) == 1
        assert (tiles[0].width, tiles[0].height) == (320, 240)

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"frame_width": 0}, "frame size"),
            ({"tile_width": 0}, "tile size"),
            ({"overlap": 1.0}, "overlap"),
            ({"overlap": -0.1}, "overlap"),
        ],
    )
    def test_invalid_parameters(self, kw: dict[str, float], match: str) -> None:
        base = {
            "frame_width": 1920,
            "frame_height": 1080,
            "tile_width": 640,
            "tile_height": 640,
            "overlap": 0.2,
        }
        with pytest.raises(ConfigError, match=match):
            generate_tiles(**{**base, **kw})  # type: ignore[arg-type]


class TestNms:
    def test_keeps_the_highest_scoring_of_a_duplicate_pair(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        kept = nms(boxes, np.array([0.7, 0.9]), iou_threshold=0.5)
        assert kept.tolist() == [1]

    def test_keeps_distinct_objects(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
        assert len(nms(boxes, np.array([0.9, 0.8]), iou_threshold=0.5)) == 2

    def test_empty_input(self) -> None:
        assert len(nms(np.empty((0, 4)), np.array([]), iou_threshold=0.5)) == 0

    def test_output_is_score_ordered(self) -> None:
        boxes = np.array(
            [[0.0, 0.0, 5.0, 5.0], [50.0, 50.0, 55.0, 55.0], [99.0, 99.0, 104.0, 104.0]]
        )
        kept = nms(boxes, np.array([0.1, 0.9, 0.5]), iou_threshold=0.5)
        assert kept.tolist() == [1, 2, 0]

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(GeometryError, match="score"):
            nms(np.array([[0.0, 0.0, 5.0, 5.0]]), np.array([0.5, 0.6]), iou_threshold=0.5)


class TestMerge:
    def test_coordinates_are_shifted_to_frame_space(self) -> None:
        tiles = [Tile(x=640, y=1280, width=640, height=640)]
        boxes, _ = merge_tile_detections(
            [np.array([[5.0, 5.0, 15.0, 25.0]])], [np.array([0.9])], tiles
        )
        assert boxes[0].tolist() == [645.0, 1285.0, 655.0, 1305.0]

    def test_seam_duplicate_is_suppressed(self) -> None:
        """One drone straddling a tile boundary is detected twice. Without dedupe it
        would count as two drones everywhere downstream."""
        tiles = [Tile(0, 0, 640, 640), Tile(512, 0, 640, 640)]
        per_tile = [
            np.array([[600.0, 100.0, 620.0, 120.0]]),
            np.array([[89.0, 100.0, 109.0, 120.0]]),
        ]
        boxes, scores = merge_tile_detections(per_tile, [np.array([0.8]), np.array([0.9])], tiles)
        assert len(boxes) == 1
        assert scores[0] == pytest.approx(0.9)
        assert boxes[0].tolist() == [601.0, 100.0, 621.0, 120.0]

    def test_distinct_objects_in_different_tiles_both_survive(self) -> None:
        tiles = [Tile(0, 0, 640, 640), Tile(640, 0, 640, 640)]
        per_tile = [np.array([[10.0, 10.0, 20.0, 20.0]]), np.array([[10.0, 400.0, 20.0, 410.0]])]
        boxes, _ = merge_tile_detections(per_tile, [np.array([0.9]), np.array([0.8])], tiles)
        assert len(boxes) == 2

    def test_tiles_with_no_detections_are_skipped_cleanly(self) -> None:
        tiles = [Tile(0, 0, 640, 640), Tile(640, 0, 640, 640)]
        per_tile = [np.empty((0, 4)), np.array([[10.0, 10.0, 20.0, 20.0]])]
        boxes, _ = merge_tile_detections(per_tile, [np.array([]), np.array([0.9])], tiles)
        assert len(boxes) == 1
        assert boxes[0].tolist() == [650.0, 10.0, 660.0, 20.0]

    def test_no_detections_anywhere(self) -> None:
        boxes, scores = merge_tile_detections(
            [np.empty((0, 4))], [np.array([])], [Tile(0, 0, 640, 640)]
        )
        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ConfigError, match="mismatched lengths"):
            merge_tile_detections([np.empty((0, 4))], [], [Tile(0, 0, 10, 10)])
