"""The job the worker executes, and how it reports back.

Two rules govern what reaches the database:

**Error messages shown to a user are sanitised.** A raw exception can carry a filesystem
path, a query fragment, or a libav string echoing file contents. `Job.error_message` gets
a short, safe sentence; the detail goes to the server-side log where the uploader cannot
read it.

**Progress written to the database is real.** `frames_processed` counts frames actually
decoded. Where the total is unknown the API reports 0 progress and the UI shows an
indeterminate state, rather than an invented percentage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tayr.config import TrackerConfig
from tayr.db.models import Job, JobStatus, TrackRecord, Video
from tayr.errors import TayrError
from tayr.security.audit import SecurityEvent, log_security_event
from tayr.security.uploads import (
    MediaProperties,
    UploadRejectedError,
    VideoLimits,
    is_safe_storage_path,
)
from tayr.worker.detector import Detector, StubDetector
from tayr.worker.pipeline import run_pipeline

_logger = logging.getLogger("tayr.worker")

# How often to persist progress, in frames. Writing every frame would make the database
# the bottleneck on a 9000-frame video.
_PROGRESS_EVERY = 30

_GENERIC_FAILURE = "processing failed; the file could not be analysed"


@dataclass(slots=True)
class _Progress:
    """Last reported progress, so a failure can still report how far it got."""

    processed: int = 0
    total: int | None = None


@dataclass(slots=True)
class JobOutcome:
    """What the job produced, for the caller to persist."""

    status: JobStatus
    frames_processed: int = 0
    frames_total: int | None = None
    error_message: str | None = None
    synthetic: bool = False
    tracks: list[TrackRecord] | None = None
    media: MediaProperties | None = None
    """Probed dimensions, so the API can show them. None when the probe failed."""


def process_video_job(
    job: Job,
    video: Video,
    *,
    storage_root: Path,
    detector: Detector | None = None,
    limits: VideoLimits | None = None,
    tracker_config: TrackerConfig | None = None,
    progress_sink: object = None,
) -> JobOutcome:
    """Run the pipeline for one job and return what should be written back.

    This function performs no database writes itself: it is pure with respect to the
    session, so it can be tested without one and so a partial failure cannot leave rows
    half-updated.
    """
    del progress_sink  # reserved for a Redis progress channel

    # StubDetector until Phase 3's detector lands. Its runs are marked synthetic and
    # that flag reaches the API response.
    detector = detector or StubDetector()

    path = storage_root / video.storage_name
    if not is_safe_storage_path(path, root=storage_root):
        # Unreachable with generated storage names; a tripwire for future changes.
        _logger.error("job %s: storage path escaped the root: %s", job.id, path)
        return JobOutcome(status=JobStatus.FAILED, error_message=_GENERIC_FAILURE)

    progress = _Progress()

    def on_progress(processed: int, total: int | None) -> None:
        progress.processed = processed
        progress.total = total

    try:
        result = run_pipeline(
            path,
            detector,
            tracker_config=tracker_config,
            limits=limits,
            progress=on_progress,
        )
    except UploadRejectedError as exc:
        # These messages are written for users and are safe to show: they say which
        # limit was exceeded and by how much.
        log_security_event(
            SecurityEvent.UPLOAD_REJECTED,
            user_id=job.owner_id,
            outcome="rejected",
            job_id=job.id,
            reason=str(exc),
        )
        return JobOutcome(
            status=JobStatus.FAILED,
            frames_processed=progress.processed,
            error_message=str(exc),
        )
    except TayrError as exc:
        _logger.exception("job %s failed", job.id)
        log_security_event(
            SecurityEvent.JOB_FAILED, user_id=job.owner_id, outcome="error", job_id=job.id
        )
        return JobOutcome(
            status=JobStatus.FAILED,
            frames_processed=progress.processed,
            error_message=str(exc),
        )
    except Exception:
        # Anything unexpected. The detail goes to the log; the user gets a generic
        # sentence, because an arbitrary exception message may carry a path or a
        # fragment of the file.
        _logger.exception("job %s failed unexpectedly", job.id)
        log_security_event(
            SecurityEvent.JOB_FAILED, user_id=job.owner_id, outcome="error", job_id=job.id
        )
        return JobOutcome(
            status=JobStatus.FAILED,
            frames_processed=progress.processed,
            error_message=_GENERIC_FAILURE,
        )

    records = [
        TrackRecord(
            job_id=job.id,
            owner_id=job.owner_id,
            track_number=track.track_id,
            # Classification lands in Phase 5. Until then the label is honestly unknown
            # rather than a guess presented as a result.
            label="unknown",
            confidence=0.0,
            first_frame=track.observed_frames[0],
            last_frame=track.observed_frames[-1],
            median_pixels_on_target=(
                float(result.features[track.track_id].median_pixels_on_target)
                if track.track_id in result.features
                else 0.0
            ),
            features_json=json.dumps(
                result.features[track.track_id].as_dict()
                if track.track_id in result.features
                else {}
            ),
        )
        for track in result.tracks
    ]

    return JobOutcome(
        status=JobStatus.SUCCEEDED,
        frames_processed=result.frames_processed,
        frames_total=result.frames_total,
        synthetic=result.synthetic,
        tracks=records,
        media=result.media,
    )


def apply_outcome(job: Job, video: Video, outcome: JobOutcome) -> list[TrackRecord]:
    """Write a JobOutcome onto the ORM objects. Returns rows for the caller to add."""
    job.status = outcome.status
    job.frames_processed = outcome.frames_processed
    job.frames_total = outcome.frames_total
    job.error_message = outcome.error_message
    job.synthetic = outcome.synthetic
    job.finished_at = datetime.now(UTC)

    # Probed properties, known only after the worker has opened the container.
    if outcome.media is not None:
        video.width = outcome.media.width
        video.height = outcome.media.height
        video.fps = outcome.media.fps
        video.duration_seconds = outcome.media.duration_seconds

    return outcome.tracks or []
