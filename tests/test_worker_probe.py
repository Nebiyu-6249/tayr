"""Probe and pipeline tests, against real video files encoded by PyAV.

These are the tests that close R1 in docs/THREAT_MODEL.md: `enforce_media_limits` was
implemented and tested from Phase 7 but nothing called it. `probe_video` calls it, and
the ordering test below proves it happens before any frame is decoded.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tayr.config import TrackerConfig
from tayr.security.uploads import UploadRejectedError, VideoLimits
from tayr.worker.detector import Detection, ScriptedDetector, StubDetector
from tayr.worker.pipeline import iter_frames, run_pipeline
from tayr.worker.probe import probe_video

av = pytest.importorskip("av", reason="PyAV is in the 'cv' extra")


def make_video(
    path: Path, *, width: int = 320, height: int = 240, fps: int = 30, n_frames: int = 60
) -> Path:
    """Encode a real MP4. Not a fixture blob - a genuine container libav must parse."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    rng = np.random.default_rng(1337)
    for i in range(n_frames):
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        # A moving bright square, so decoded frames are not uniform.
        x = 10 + (i * 3) % max(1, width - 30)
        img[100:120, x : x + 20] = 220
        img = np.clip(img.astype(np.int16) + rng.integers(-4, 5, img.shape), 0, 255).astype(
            np.uint8
        )
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "clip.mp4")


class TestProbe:
    def test_reports_real_properties(self, video: Path) -> None:
        props = probe_video(video)
        assert props.width == 320
        assert props.height == 240
        assert props.fps == pytest.approx(30.0, abs=0.5)
        assert props.duration_seconds == pytest.approx(2.0, abs=0.3)

    def test_rejects_a_non_video_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "not-a-video.mp4"
        bad.write_bytes(b"this is definitely not a video file" * 10)
        with pytest.raises(UploadRejectedError, match="could not read"):
            probe_video(bad)

    def test_error_message_does_not_echo_file_contents(self, tmp_path: Path) -> None:
        """A libav error string can quote bytes from the file back to the uploader."""
        secret = b"SENSITIVE-MARKER-abcdef0123456789"
        bad = tmp_path / "x.mp4"
        bad.write_bytes(secret * 20)
        with pytest.raises(UploadRejectedError) as exc:
            probe_video(bad)
        assert b"SENSITIVE-MARKER".decode() not in str(exc.value)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(UploadRejectedError, match="not a file"):
            probe_video(tmp_path / "nope.mp4")

    def test_enforces_resolution_limit(self, video: Path) -> None:
        with pytest.raises(UploadRejectedError, match="resolution"):
            probe_video(video, limits=VideoLimits(max_width=100, max_height=100))

    def test_enforces_duration_limit(self, video: Path) -> None:
        with pytest.raises(UploadRejectedError, match="duration"):
            probe_video(video, limits=VideoLimits(max_duration_seconds=0.5))

    def test_enforces_fps_limit(self, video: Path) -> None:
        with pytest.raises(UploadRejectedError, match="frame rate"):
            probe_video(video, limits=VideoLimits(max_fps=10.0))

    def test_enforces_total_pixel_limit(self, video: Path) -> None:
        """The decompression-bomb cap, now actually reachable from a real file."""
        with pytest.raises(UploadRejectedError, match="decompression bomb"):
            probe_video(video, limits=VideoLimits(max_total_pixels=1000.0))


class TestLimitsPrecedeDecoding:
    """R1's actual requirement: bounding happens BEFORE decoding, not after."""

    def test_pipeline_refuses_before_decoding_a_single_frame(
        self, video: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decoded: list[int] = []

        def spy(*_args: object, **_kwargs: object) -> Iterator[npt.NDArray[np.uint8]]:
            decoded.append(1)
            yield from ()

        monkeypatch.setattr("tayr.worker.pipeline.iter_frames", spy)
        with pytest.raises(UploadRejectedError):
            run_pipeline(video, StubDetector(), limits=VideoLimits(max_width=64, max_height=64))
        assert decoded == [], "decoding started despite the media exceeding a limit"


class TestFrameIteration:
    def test_yields_rgb_frames_of_the_right_shape(self, video: Path) -> None:
        frames = list(iter_frames(video, max_frames=5))
        assert len(frames) == 5
        assert frames[0].shape == (240, 320, 3)
        assert frames[0].dtype == np.uint8

    def test_max_frames_bounds_the_decode(self, video: Path) -> None:
        assert len(list(iter_frames(video, max_frames=3))) == 3

    def test_streams_rather_than_materialising(self, video: Path) -> None:
        """A generator, not a list: a 300s 4K video decoded whole is hundreds of GB."""
        frames = iter_frames(video)
        assert next(frames) is not None
        # Generator, not a list: closing mid-iteration must release the container.
        assert isinstance(frames, Generator)
        frames.close()


class TestPipeline:
    def test_runs_end_to_end(self, video: Path) -> None:
        result = run_pipeline(video, StubDetector())
        assert result.frames_processed > 0
        assert result.media.width == 320

    def test_progress_is_real(self, video: Path) -> None:
        """Counts frames actually decoded, against a total derived from the probe."""
        seen: list[tuple[int, int | None]] = []
        result = run_pipeline(video, StubDetector(), progress=lambda p, t: seen.append((p, t)))
        assert len(seen) == result.frames_processed
        assert [p for p, _ in seen] == list(range(1, result.frames_processed + 1))
        assert seen[-1][1] is not None and seen[-1][1] > 0

    def test_stub_detector_finds_nothing_rather_than_inventing_boxes(self, video: Path) -> None:
        """Fabricated detections would flow into tracks, features and a report, and be
        indistinguishable from real ones by the time anyone looked."""
        result = run_pipeline(video, StubDetector())
        assert result.n_tracks == 0

    def test_stub_run_is_marked_synthetic(self, video: Path) -> None:
        result = run_pipeline(video, StubDetector())
        assert result.synthetic is True
        assert any("SYNTHETIC" in n for n in result.notes)

    def test_scripted_detections_produce_tracks_and_features(self, video: Path) -> None:
        """Exercises tracking and feature extraction against known input, so a pipeline
        bug is distinguishable from a detector bug."""
        per_frame = []
        for i in range(60):
            cx, cy, side = 50.0 + i * 2.0, 100.0, 24.0
            box = [[cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]]
            per_frame.append(
                Detection(np.array(box, dtype=np.float64), np.array([0.9], dtype=np.float64))
            )
        result = run_pipeline(
            video,
            ScriptedDetector(per_frame),
            tracker_config=TrackerConfig(min_hits=3, max_age=10),
        )
        assert result.n_tracks == 1
        track_id = result.tracks[0].track_id
        assert track_id in result.features
        assert result.features[track_id].mean_speed == pytest.approx(2.0, abs=0.2)
        assert result.synthetic is True

    def test_short_tracks_get_no_features_and_are_reported(self, video: Path) -> None:
        """A track too short to support an acceleration variance gets no features, and
        that omission is stated rather than silently producing zeros."""
        empty = Detection(np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64))
        # The target appears only in the last few frames, so the track is still alive
        # when the video ends. A track that instead died mid-video would be removed
        # entirely, which is a different case from "kept but too short".
        per_frame = [empty for _ in range(55)]
        per_frame += [
            Detection(
                np.array([[10.0 + i, 10.0, 30.0 + i, 30.0]], dtype=np.float64),
                np.array([0.9], dtype=np.float64),
            )
            for i in range(5)
        ]
        result = run_pipeline(
            video, ScriptedDetector(per_frame), tracker_config=TrackerConfig(min_hits=2, max_age=5)
        )
        assert result.n_tracks == 1
        assert result.tracks[0].n_observations == 5
        assert result.features == {}
        assert any("fewer than" in n for n in result.notes)

    def test_max_frames_is_respected(self, video: Path) -> None:
        result = run_pipeline(video, StubDetector(), max_frames=10)
        assert result.frames_processed == 10
        assert result.frames_total == 10
