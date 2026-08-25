"""Census tests.

The census answers whether the hypothesis is testable at all, so its counting must be
right and its warnings must fire. A census that silently over-counts tracks would let
an under-powered experiment look adequately powered.
"""

from __future__ import annotations

import json

from tayr.datasets.census import take_census
from tayr.datasets.converters.anti_uav import parse_anti_uav_json
from tayr.datasets.converters.drone_vs_bird import parse_drone_vs_bird
from tayr.datasets.schema import (
    BoxAnnotation,
    DatasetAnnotation,
    FrameAnnotation,
    ObjectClass,
    TrackIdSource,
    VideoAnnotation,
)


def _tracked_video(
    name: str, label: ObjectClass, track_id: int, n: int, side: float = 20.0
) -> VideoAnnotation:
    frames = tuple(
        FrameAnnotation(
            frame_index=i,
            objects=(BoxAnnotation(0.0, 0.0, side, side, label, track_id),),
        )
        for i in range(n)
    )
    return VideoAnnotation(
        source_video=name, frames=frames, track_id_source=TrackIdSource.ANNOTATED
    )


class TestCounting:
    def test_counts_tracks_not_boxes(self) -> None:
        """The whole point: 300 boxes across 3 videos is 3 tracks, not 300 samples."""
        ds = DatasetAnnotation(
            name="t",
            split="train",
            videos=tuple(
                _tracked_video(f"v{i}", ObjectClass.DRONE, track_id=0, n=100) for i in range(3)
            ),
        )
        c = take_census(ds)
        assert c.n_boxes == 300
        assert c.n_tracks == 3
        assert c.tracks_per_class["drone"] == 3

    def test_track_key_is_video_plus_id_not_id_alone(self) -> None:
        """Two videos each with track_id=0 are two tracks, not one."""
        ds = DatasetAnnotation(
            name="t",
            split="train",
            videos=(
                _tracked_video("a", ObjectClass.DRONE, track_id=0, n=5),
                _tracked_video("b", ObjectClass.DRONE, track_id=0, n=5),
            ),
        )
        assert take_census(ds).n_tracks == 2

    def test_counts_empty_frames(self) -> None:
        ds = DatasetAnnotation(
            name="t",
            split="train",
            videos=(parse_drone_vs_bird("0 1 10 10 5 5 drone\n1 0\n2 0\n", source_video="v"),),
        )
        c = take_census(ds)
        assert c.n_frames == 3
        assert c.n_empty_frames == 2

    def test_size_buckets_reflect_box_sizes(self) -> None:
        ds = DatasetAnnotation(
            name="t",
            split="train",
            videos=(
                _tracked_video("tiny", ObjectClass.DRONE, 0, n=1, side=4.0),
                _tracked_video("large", ObjectClass.DRONE, 0, n=1, side=64.0),
            ),
        )
        c = take_census(ds)
        assert c.size_buckets["<8px"] == 1
        assert c.size_buckets[">32px"] == 1

    def test_track_span_and_median_size(self) -> None:
        ds = DatasetAnnotation(
            name="t", split="train", videos=(_tracked_video("v", ObjectClass.BIRD, 3, n=10),)
        )
        track = take_census(ds).tracks[0]
        assert track.n_observations == 10
        assert track.first_frame == 0
        assert track.last_frame == 9
        assert track.span == 10
        assert track.median_pixels_on_target == 20.0


class TestWarnings:
    def test_warns_when_a_dataset_has_zero_tracks(self) -> None:
        """Drone-vs-Bird has no track ids, so it trains a detector and contributes
        nothing to the motion hypothesis. That must be stated, not discovered later."""
        ds = DatasetAnnotation(
            name="dvb",
            split="train",
            videos=(parse_drone_vs_bird("0 1 10 10 5 5 drone\n", source_video="v"),),
        )
        c = take_census(ds)
        assert c.n_tracks == 0
        assert any("ZERO tracks" in w for w in c.warnings)
        assert any("no track identity" in w for w in c.warnings)

    def test_warns_when_track_identity_was_inferred(self) -> None:
        video, _ = parse_anti_uav_json(
            json.dumps({"gt_rect": [[1, 1, 9, 9]], "exist": [1]}), source_video="s"
        )
        c = take_census(DatasetAnnotation(name="antiuav", split="test", videos=(video,)))
        assert any("INFERRED" in w for w in c.warnings)

    def test_warns_on_thin_per_class_track_counts(self) -> None:
        ds = DatasetAnnotation(
            name="t", split="train", videos=(_tracked_video("v", ObjectClass.BIRD, 0, n=5),)
        )
        warnings = take_census(ds).warnings
        assert any("bird track(s)" in w and "overclaim" in w for w in warnings)
        assert any("0 aircraft track(s)" in w for w in warnings)

    def test_warns_when_a_track_changes_class_midway(self) -> None:
        """A label that flips mid-track is an annotation or association bug, and the
        majority vote used to resolve it must be announced."""
        frames = (
            FrameAnnotation(0, (BoxAnnotation(0, 0, 9, 9, ObjectClass.DRONE, 1),)),
            FrameAnnotation(1, (BoxAnnotation(0, 0, 9, 9, ObjectClass.DRONE, 1),)),
            FrameAnnotation(2, (BoxAnnotation(0, 0, 9, 9, ObjectClass.BIRD, 1),)),
        )
        ds = DatasetAnnotation(
            name="t",
            split="train",
            videos=(VideoAnnotation("v", frames, track_id_source=TrackIdSource.ANNOTATED),),
        )
        c = take_census(ds)
        assert any("different class labels" in w for w in c.warnings)
        assert c.tracks[0].label is ObjectClass.DRONE  # majority


class TestRender:
    def test_render_includes_the_numbers_that_matter(self) -> None:
        ds = DatasetAnnotation(
            name="t", split="train", videos=(_tracked_video("v", ObjectClass.DRONE, 0, n=4),)
        )
        out = take_census(ds).render()
        assert "TRACKS per class" in out
        assert "pixels-on-target distribution" in out
        assert "WARNINGS" in out
