"""Converter tests.

Per `.claude/skills/dataset-converter`, every converter needs a round-trip AND
hand-computed absolute values (a round-trip alone passes when both directions share the
same sign error), plus: a frame with zero objects, a frame with several, the smallest
box in the dataset, and an absent-target frame where the format has one.

All fixtures are written inline. No annotation file from any dataset is copied into
this repository - Drone-vs-Bird grants no redistribution rights, and converted
annotations are text that the CI licence-guard would not catch.
"""

from __future__ import annotations

import json

import pytest

from tayr.datasets.converters.anti_uav import parse_anti_uav_json, write_anti_uav_json
from tayr.datasets.converters.drone_vs_bird import (
    parse_drone_vs_bird,
    write_drone_vs_bird,
)
from tayr.datasets.converters.dut_anti_uav import parse_dut_tracking_gt_first
from tayr.datasets.schema import ObjectClass, TrackIdSource
from tayr.errors import ConfigError

# framenum num_objs x_left y_top w h class
DVB_SAMPLE = """\
0 1 100 200 30 40 drone
1 0
2 2 100 200 30 40 drone 500 600 4 4 bird
3 1 105 205 28 38 drone
"""


class TestDroneVsBird:
    def test_round_trip_is_textually_exact(self) -> None:
        video = parse_drone_vs_bird(DVB_SAMPLE, source_video="v1")
        assert write_drone_vs_bird(video) == DVB_SAMPLE

    def test_absolute_values_hand_computed(self) -> None:
        """xywh(100,200,30,40) must become xyxy(100,200,130,240) - not 100,200,30,40
        and not 100,200,230,440. A round-trip alone would not catch either."""
        video = parse_drone_vs_bird(DVB_SAMPLE, source_video="v1")
        obj = video.frames[0].objects[0]
        assert (obj.x1, obj.y1, obj.x2, obj.y2) == (100.0, 200.0, 130.0, 240.0)
        assert obj.label is ObjectClass.DRONE

    def test_frame_with_zero_objects_is_kept_not_dropped(self) -> None:
        """Empty frames are the false-alarm denominator. Dropping them inflates the
        headline safety metric."""
        video = parse_drone_vs_bird(DVB_SAMPLE, source_video="v1")
        assert len(video.frames) == 4
        assert video.frames[1].is_empty
        assert video.n_empty_frames == 1

    def test_frame_with_multiple_objects(self) -> None:
        video = parse_drone_vs_bird(DVB_SAMPLE, source_video="v1")
        frame = video.frames[2]
        assert len(frame.objects) == 2
        assert [o.label for o in frame.objects] == [ObjectClass.DRONE, ObjectClass.BIRD]

    def test_smallest_box_survives_conversion(self) -> None:
        """A 4x4 box must not come out as zero-width. This is a small-object project;
        that is the bug worth testing for."""
        video = parse_drone_vs_bird(DVB_SAMPLE, source_video="v1")
        bird = video.frames[2].objects[1]
        assert bird.x2 - bird.x1 == 4.0
        assert bird.y2 - bird.y1 == 4.0

    def test_no_track_identity_is_recorded_honestly(self) -> None:
        """The format has no track ids. Claiming otherwise would let the census count
        tracks that do not exist."""
        video = parse_drone_vs_bird(DVB_SAMPLE, source_video="v1")
        assert video.track_id_source is TrackIdSource.NONE
        assert all(o.track_id is None for f in video.frames for o in f.objects)

    def test_unknown_class_token_is_not_coerced_to_drone(self) -> None:
        video = parse_drone_vs_bird("0 1 10 10 5 5 quadcopter\n", source_video="v")
        assert video.frames[0].objects[0].label is ObjectClass.UNKNOWN

    def test_frames_are_sorted_and_base_recorded(self) -> None:
        video = parse_drone_vs_bird("7 0\n5 0\n6 0\n", source_video="v")
        assert [f.frame_index for f in video.frames] == [5, 6, 7]
        assert video.frame_index_base == 5

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("0 2 10 10 5 5 drone\n", "expected 12 fields"),
            ("0 1 10 10 5 5 drone extra\n", "expected 7 fields"),
            ("abc 1 10 10 5 5 drone\n", "must be integers"),
            ("0 -1\n", "negative object count"),
            ("0 1 10 10 0 5 drone\n", "non-positive size"),
            ("0\n", "at least"),
        ],
    )
    def test_malformed_lines_raise_with_line_number(self, text: str, match: str) -> None:
        """A malformed line is never skipped: silently dropping it shrinks the dataset
        and changes every metric computed from it."""
        with pytest.raises(ConfigError, match=match):
            parse_drone_vs_bird(text, source_video="v")


class TestAntiUav:
    def test_round_trip(self) -> None:
        native = json.dumps(
            {"gt_rect": [[10, 20, 30, 40], [], [12, 22, 30, 40]], "exist": [1, 0, 1]}
        )
        video, _ = parse_anti_uav_json(native, source_video="seq1")
        assert json.loads(write_anti_uav_json(video)) == json.loads(native)

    def test_absolute_values_left_top_width_height(self) -> None:
        """Verified convention is (left, top, width, height), so [10,20,30,40] is
        xyxy(10,20,40,60) - NOT xyxy(10,20,30,40)."""
        video, _ = parse_anti_uav_json(
            json.dumps({"gt_rect": [[10, 20, 30, 40]], "exist": [1]}), source_video="s"
        )
        obj = video.frames[0].objects[0]
        assert (obj.x1, obj.y1, obj.x2, obj.y2) == (10.0, 20.0, 40.0, 60.0)

    def test_exist_flag_is_honoured(self) -> None:
        """A converter ignoring the exist flag emits boxes for absent targets and
        poisons training."""
        video, report = parse_anti_uav_json(
            json.dumps({"gt_rect": [[10, 20, 30, 40], [50, 60, 5, 5]], "exist": [1, 0]}),
            source_video="s",
        )
        assert report.used_exist_key is True
        assert report.n_present == 1
        assert video.frames[1].is_empty  # box present in file, but exist=0

    def test_falls_back_to_empty_rect_when_no_exist_key(self) -> None:
        """The Anti-UAV410 loader reads no `exist` key, so the fallback must work."""
        _video, report = parse_anti_uav_json(
            json.dumps({"gt_rect": [[10, 20, 30, 40], [], [0, 0, 0, 0]]}), source_video="s"
        )
        assert report.used_exist_key is False
        assert report.n_present == 1
        assert report.n_absent == 2

    def test_absent_frames_are_preserved_in_order(self) -> None:
        video, _ = parse_anti_uav_json(
            json.dumps({"gt_rect": [[], [10, 10, 4, 4], []], "exist": [0, 1, 0]}),
            source_video="s",
        )
        assert [f.frame_index for f in video.frames] == [0, 1, 2]
        assert [f.is_empty for f in video.frames] == [True, False, True]

    def test_track_identity_is_marked_inferred(self) -> None:
        """Single-target inference is an assumption and must be visible as one."""
        video, _ = parse_anti_uav_json(
            json.dumps({"gt_rect": [[1, 1, 5, 5]], "exist": [1]}), source_video="s", track_id=7
        )
        assert video.track_id_source is TrackIdSource.SINGLE_TARGET
        assert video.frames[0].objects[0].track_id == 7

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ("not json", "not valid JSON"),
            (json.dumps([1, 2]), "expected a JSON object"),
            (json.dumps({"boxes": []}), "no 'gt_rect' key"),
            (json.dumps({"gt_rect": "nope"}), "must be a list"),
            (json.dumps({"gt_rect": [[1, 1, 1, 1]], "exist": [1, 1]}), "one-to-one"),
            (json.dumps({"gt_rect": [[1, 2, 3]], "exist": [1]}), "expected 4 numbers"),
            (json.dumps({"gt_rect": [[1, 2, 0, 4]], "exist": [1]}), "non-positive size"),
        ],
    )
    def test_malformed_input_raises(self, payload: str, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            parse_anti_uav_json(payload, source_video="s")

    def test_multi_object_frame_cannot_be_serialised(self) -> None:
        """The native format is single-target; refusing is better than writing a file
        that silently loses objects."""
        from tayr.datasets.schema import BoxAnnotation, FrameAnnotation, VideoAnnotation

        video = VideoAnnotation(
            source_video="s",
            frames=(
                FrameAnnotation(
                    frame_index=0,
                    objects=(
                        BoxAnnotation(0, 0, 5, 5, ObjectClass.DRONE, 0),
                        BoxAnnotation(9, 9, 14, 14, ObjectClass.DRONE, 1),
                    ),
                ),
            ),
            track_id_source=TrackIdSource.SINGLE_TARGET,
        )
        with pytest.raises(ConfigError, match="single-target"):
            write_anti_uav_json(video)


class TestDutAntiUav:
    def test_the_tracking_subset_raises_rather_than_faking_tracks(self) -> None:
        """It ships one first-frame box per video: 20 boxes, 24,804 frames, no tracks.

        Returning 20 one-frame "tracks" would present a data gap as a conversion result.
        The detection subset is a different shape and is supported as `voc`.
        """
        with pytest.raises(NotImplementedError, match="only a first-frame box"):
            parse_dut_tracking_gt_first("whatever", source_video="v")

    def test_the_detection_subset_is_reachable_as_voc(self) -> None:
        from tayr.datasets.loader import SUPPORTED_FORMATS

        assert "voc" in SUPPORTED_FORMATS
