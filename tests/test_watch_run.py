"""The live watch path, and the one property it exists to protect.

`synthetic` is derived from `detector.is_real` and travels from there into the pipeline
result, the tool context, the decision record, the manifest and the notification. The
failure this guards against is a placeholder run reported as a real one, and that failure
is always a human forgetting a flag - so the tests here check that no flag exists to
forget, not merely that the current call sites pass the right value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tayr.config import Config, DetectorConfig
from tayr.detection.factory import (
    TRAINED_ON_SYNTHETIC,
    DetectorChoice,
    _training_provenance,
    build_detector,
)
from tayr.errors import ConfigError
from tayr.worker.detector import ScriptedDetector, StubDetector
from tests.cv_extra import requires_cv_extra

REPO = Path(__file__).resolve().parents[1]
SITES = REPO / "configs" / "sites" / "demo.yaml"


class TestDetectorChoice:
    def test_a_stub_is_never_a_silent_fallback(self) -> None:
        """Handing back a stand-in would satisfy the type and find nothing."""
        with pytest.raises(ConfigError, match="Refusing to substitute a stand-in"):
            build_detector(DetectorConfig(), device="cpu", allow_stub=False)

    def test_the_error_names_the_flag_that_fixes_it(self) -> None:
        with pytest.raises(ConfigError, match="--checkpoint"):
            build_detector(DetectorConfig(), device="cpu", allow_stub=False)

    def test_an_explicitly_allowed_stub_is_labelled(self) -> None:
        choice = build_detector(DetectorConfig(), device="cpu", allow_stub=True)
        assert choice.synthetic is True
        assert any("NO TRAINED DETECTOR" in note for note in choice.notes)

    def test_synthetic_is_derived_from_the_detector_not_stored(self) -> None:
        """There is no field to set wrongly: it is computed on every read."""
        assert DetectorChoice(detector=StubDetector(), checkpoint_sha256=None).synthetic
        assert DetectorChoice(detector=ScriptedDetector([]), checkpoint_sha256=None).synthetic

    def test_an_unimplemented_backend_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="no implementation"):
            build_detector(DetectorConfig(backend="dfine"), device="cpu", allow_stub=True)


class TestTrainingProvenance:
    """`synthetic` answers "was the detector real". It does not answer "were the weights
    fitted to real footage", and a model trained on placeholder data produces real
    detections describing nothing. CLAUDE.md 1.3 wants that said, not inferred."""

    def note(self, tmp_path: Path, manifest: object | None) -> str:
        checkpoint = tmp_path / "checkpoint_best_total.pth"
        checkpoint.write_bytes(b"not a real checkpoint")
        if manifest is not None:
            (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return _training_provenance(checkpoint)

    def test_synthetic_training_data_is_reported_loudly(self, tmp_path: Path) -> None:
        note = self.note(tmp_path, {"run_id": "voc-02", "git_commit": "a" * 40, "synthetic": True})
        assert note.startswith(TRAINED_ON_SYNTHETIC)
        assert "voc-02" in note

    def test_real_training_data_names_the_run_that_produced_it(self, tmp_path: Path) -> None:
        note = self.note(tmp_path, {"run_id": "t4-01", "git_commit": "b" * 40, "synthetic": False})
        assert not note.startswith(TRAINED_ON_SYNTHETIC)
        assert "t4-01" in note

    def test_a_missing_manifest_is_unknown_not_assumed_fine(self, tmp_path: Path) -> None:
        """An absent manifest is not evidence of real training data."""
        assert "UNKNOWN" in self.note(tmp_path, None)

    def test_an_unparseable_manifest_says_so_rather_than_claiming_a_check(
        self, tmp_path: Path
    ) -> None:
        checkpoint = tmp_path / "checkpoint_best_total.pth"
        checkpoint.write_bytes(b"x")
        (tmp_path / "manifest.json").write_text("{ truncated", encoding="utf-8")
        assert "UNREADABLE" in _training_provenance(checkpoint)

    def test_a_missing_synthetic_key_is_not_read_as_true(self, tmp_path: Path) -> None:
        """`.get("synthetic") is True` and not a truthiness test: absent means absent."""
        note = self.note(tmp_path, {"run_id": "old", "git_commit": "c" * 40})
        assert not note.startswith(TRAINED_ON_SYNTHETIC)


@requires_cv_extra
class TestWatchRun:
    """End to end on a tiny video, with a stand-in detector.

    The stub is the point: it makes `synthetic` True, which is the case that must stay
    visible. The real-detector case is exercised by hand against a trained checkpoint -
    a checkpoint is too large to keep in the repository, and generating one in a test
    would take minutes.
    """

    def video(self, tmp_path: Path) -> Path:
        import av

        path = tmp_path / "clip.mp4"
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(40):
            frame = np.full((240, 320, 3), 40, np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return path

    def run(self, tmp_path: Path) -> object:
        from tayr.agent.live import run_watch

        return run_watch(
            self.video(tmp_path),
            Config.model_validate({"device": "cpu"}),
            output_dir=tmp_path / "out",
            site_registry_path=SITES,
            repo=REPO,
            run_id="test-watch",
            allow_stub=True,
        )

    def test_a_stub_run_is_synthetic_in_every_artifact(self, tmp_path: Path) -> None:
        run = self.run(tmp_path)
        assert run.synthetic is True  # type: ignore[attr-defined]
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["synthetic"] is True
        assert any("SYNTHETIC DATA" in note for note in manifest["notes"])

    def test_the_device_is_recorded_rather_than_assumed(self, tmp_path: Path) -> None:
        run = self.run(tmp_path)
        assert run.device.device == "cpu"  # type: ignore[attr-defined]
        assert any("device:" in note for note in run.notes)  # type: ignore[attr-defined]

    def test_a_missing_video_is_named(self, tmp_path: Path) -> None:
        from tayr.agent.live import run_watch

        with pytest.raises(ConfigError, match="video not found"):
            run_watch(
                tmp_path / "nope.mp4",
                Config.model_validate({"device": "cpu"}),
                output_dir=tmp_path / "out",
                site_registry_path=SITES,
                allow_stub=True,
            )

    def test_an_unknown_site_is_refused(self, tmp_path: Path) -> None:
        from tayr.agent.live import run_watch

        with pytest.raises(ConfigError, match="is not in"):
            run_watch(
                self.video(tmp_path),
                Config.model_validate({"device": "cpu"}),
                output_dir=tmp_path / "out",
                site_registry_path=SITES,
                site_id="nowhere",
                allow_stub=True,
            )


class TestATrainedDetectorIsNotATrainedClassifier:
    """Wiring a real detector must not imply the classifier is fitted.

    `no_classifier_trained` is a different claim from "the classifier was unsure", they
    drive different rules, and a detector that finds drones says nothing about whether
    one is a bird.
    """

    def test_the_snapshot_default_is_untrained(self) -> None:
        from tayr.agent.tools import TrackSnapshot

        snapshot = TrackSnapshot(
            track_id="t",
            job_id="j",
            track_number=1,
            first_frame=0,
            last_frame=10,
            n_observations=11,
            median_pixels_on_target=20.0,
            fps=30.0,
        )
        assert snapshot.classifier_trained is False
        assert snapshot.classifier_label == "unknown"

    def test_the_live_path_never_sets_it_true(self) -> None:
        """Read the source: a future edit that wires the detector into this flag fails."""
        source = (REPO / "src" / "tayr" / "agent" / "live.py").read_text(encoding="utf-8")
        assert "classifier_trained=False" in source
        assert "classifier_trained=True" not in source


class TestNegativeFootage:
    def make_video(self, path: Path, *, frames: int = 20, fps: int = 25) -> Path:
        import av

        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(frames):
            frame = np.full((240, 320, 3), 60, np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return path

    def test_an_empty_directory_is_refused(self, tmp_path: Path) -> None:
        from tayr.eval.harness import find_negative_footage

        with pytest.raises(ConfigError, match="nothing to measure"):
            find_negative_footage(tmp_path)

    def test_loose_frames_are_found(self, tmp_path: Path) -> None:
        from tayr.eval.harness import find_negative_footage

        (tmp_path / "a.jpg").write_bytes(b"x")
        found = find_negative_footage(tmp_path)
        assert found.is_video is False
        assert len(found.images) == 1

    @requires_cv_extra
    def test_videos_win_over_stray_images_and_the_choice_is_stated(self, tmp_path: Path) -> None:
        """Mixing them gives a frame count with no single duration behind it."""
        from tayr.eval.harness import find_negative_footage

        self.make_video(tmp_path / "clip.mp4")
        (tmp_path / "screenshot.jpg").write_bytes(b"x")
        found = find_negative_footage(tmp_path)
        assert found.is_video is True
        assert found.images == ()
        assert any("were ignored" in note for note in found.notes)

    def test_loose_frames_without_an_fps_are_refused(self, tmp_path: Path) -> None:
        """A per-hour rate is frames/fps/3600, so a guessed fps scales the headline."""
        from tayr.eval.harness import measure_false_alarms

        (tmp_path / "a.jpg").write_bytes(b"x")
        with pytest.raises(ConfigError, match="no container to read a frame rate from"):
            measure_false_alarms(StubDetector(), tmp_path, fps=None, confidence_threshold=0.25)

    @requires_cv_extra
    def test_a_videos_frame_rate_comes_from_its_container(self, tmp_path: Path) -> None:
        from tayr.eval.harness import measure_false_alarms

        self.make_video(tmp_path / "clip.mp4", frames=25, fps=25)
        rate, notes = measure_false_alarms(
            StubDetector(), tmp_path, fps=None, confidence_threshold=0.25
        )
        assert rate.n_frames == 25
        assert rate.duration_hours == pytest.approx(1.0 / 3600, rel=0.05)
        assert any("25.000 fps" in note for note in notes)

    @requires_cv_extra
    def test_a_configured_fps_is_ignored_for_videos_and_said_so(self, tmp_path: Path) -> None:
        """The container's value is the one that is actually true."""
        from tayr.eval.harness import measure_false_alarms

        self.make_video(tmp_path / "clip.mp4", frames=25, fps=25)
        _, notes = measure_false_alarms(
            StubDetector(), tmp_path, fps=999.0, confidence_threshold=0.25
        )
        assert any("was ignored" in note for note in notes)

    def test_the_config_no_longer_demands_an_fps_it_can_read(self) -> None:
        """Videos carry their own frame rate; only loose frames need one declared."""
        Config.model_validate({"eval": {"negatives_dir": "/tmp"}})  # noqa: S108
        with pytest.raises(ValueError, match="negatives_fps is set but"):
            Config.model_validate({"eval": {"negatives_fps": 30.0}})
