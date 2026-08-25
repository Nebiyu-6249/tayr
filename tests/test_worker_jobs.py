"""Job handler tests.

Two behaviours matter beyond the happy path: a failure must not leak internal detail to
the uploader, and a run using a stand-in detector must be marked synthetic all the way
through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tayr.db.models import Job, JobStatus, Video
from tayr.security.uploads import VideoLimits
from tayr.worker.detector import StubDetector
from tayr.worker.jobs import apply_outcome, process_video_job

pytest.importorskip("av", reason="PyAV is in the 'cv' extra")

from tests.test_worker_probe import make_video


@pytest.fixture
def storage(tmp_path: Path) -> Path:
    root = tmp_path / "storage"
    root.mkdir()
    return root


def _job_and_video(storage: Path, name: str = "clip.mp4") -> tuple[Job, Video]:
    make_video(storage / name)
    video = Video(
        id="video-1",
        owner_id="user-1",
        storage_name=name,
        original_filename="holiday.mp4",
        container="mp4",
        size_bytes=(storage / name).stat().st_size,
    )
    job = Job(id="job-1", owner_id="user-1", video_id="video-1", status=JobStatus.QUEUED)
    return job, video


class TestSuccess:
    def test_succeeds_and_reports_real_frame_counts(self, storage: Path) -> None:
        job, video = _job_and_video(storage)
        outcome = process_video_job(job, video, storage_root=storage)
        assert outcome.status is JobStatus.SUCCEEDED
        assert outcome.frames_processed > 0
        assert outcome.frames_total is not None

    def test_stub_detector_marks_the_job_synthetic(self, storage: Path) -> None:
        """The flag must reach the database, and from there the API response."""
        job, video = _job_and_video(storage)
        outcome = process_video_job(job, video, storage_root=storage, detector=StubDetector())
        assert outcome.synthetic is True
        apply_outcome(job, video, outcome)
        assert job.synthetic is True

    def test_apply_outcome_writes_terminal_state(self, storage: Path) -> None:
        job, video = _job_and_video(storage)
        outcome = process_video_job(job, video, storage_root=storage)
        apply_outcome(job, video, outcome)
        assert job.status is JobStatus.SUCCEEDED
        assert job.finished_at is not None
        assert job.error_message is None


class TestFailureHandling:
    def test_limit_violation_reports_which_limit(self, storage: Path) -> None:
        """These messages are written for users and are safe to show."""
        job, video = _job_and_video(storage)
        outcome = process_video_job(
            job, video, storage_root=storage, limits=VideoLimits(max_width=64, max_height=64)
        )
        assert outcome.status is JobStatus.FAILED
        assert outcome.error_message is not None
        assert "resolution" in outcome.error_message

    def test_corrupt_file_does_not_leak_its_contents(self, storage: Path) -> None:
        """A libav error string can echo bytes from the file back to the uploader."""
        marker = "SENSITIVE-MARKER-9f8e7d6c"
        (storage / "bad.mp4").write_text(marker * 100, encoding="utf-8")
        video = Video(
            id="v",
            owner_id="u",
            storage_name="bad.mp4",
            original_filename="x.mp4",
            container="mp4",
            size_bytes=10,
        )
        job = Job(id="j", owner_id="u", video_id="v", status=JobStatus.QUEUED)
        outcome = process_video_job(job, video, storage_root=storage)
        assert outcome.status is JobStatus.FAILED
        assert marker not in (outcome.error_message or "")

    def test_missing_file_fails_cleanly(self, storage: Path) -> None:
        video = Video(
            id="v",
            owner_id="u",
            storage_name="absent.mp4",
            original_filename="x.mp4",
            container="mp4",
            size_bytes=10,
        )
        job = Job(id="j", owner_id="u", video_id="v", status=JobStatus.QUEUED)
        outcome = process_video_job(job, video, storage_root=storage)
        assert outcome.status is JobStatus.FAILED
        assert outcome.error_message

    def test_error_message_never_contains_a_filesystem_path(self, storage: Path) -> None:
        """An absolute path tells an attacker the server's directory layout."""
        video = Video(
            id="v",
            owner_id="u",
            storage_name="absent.mp4",
            original_filename="x.mp4",
            container="mp4",
            size_bytes=10,
        )
        job = Job(id="j", owner_id="u", video_id="v", status=JobStatus.QUEUED)
        outcome = process_video_job(job, video, storage_root=storage)
        assert str(storage) not in (outcome.error_message or "")


class TestTrackPersistence:
    def test_labels_are_unknown_until_the_classifier_exists(self, storage: Path) -> None:
        """Phase 5 has not run. A guessed label presented as a result would be exactly
        the fabrication this project's rules forbid."""
        job, video = _job_and_video(storage)
        outcome = process_video_job(job, video, storage_root=storage)
        for record in outcome.tracks or []:
            assert record.label == "unknown"
            assert record.confidence == 0.0

    def test_track_rows_carry_ownership(self, storage: Path) -> None:
        """owner_id on the row is what row-level security filters on."""
        job, video = _job_and_video(storage)
        outcome = process_video_job(job, video, storage_root=storage)
        for record in outcome.tracks or []:
            assert record.owner_id == job.owner_id
            assert record.job_id == job.id
