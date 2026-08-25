"""arq worker entrypoint: pulls jobs from Redis and runs the pipeline.

Closes R1b. The consumer owns three responsibilities the pipeline deliberately does not:
opening a database session, writing progress, and recording a terminal state even when
the job raises.

**A failure must still write a terminal state.** A job left as `running` after the
process dies is indistinguishable from one still working, and the user waits forever.
Every exit path here writes `succeeded` or `failed`.

**The worker holds its own credentials.** It connects as its own database role, scoped
to what a job needs, so compromising the decoder does not yield database-wide access.
See docs/SECURITY.md.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tayr.db.models import Job, JobStatus, Video
from tayr.security.audit import SecurityEvent, log_security_event
from tayr.worker.jobs import apply_outcome, process_video_job

_logger = logging.getLogger("tayr.worker")

# How often to persist progress. Writing every frame would make the database the
# bottleneck on a 9000-frame video.
PROGRESS_INTERVAL_FRAMES = 30


async def process_job(ctx: dict[str, Any], job_id: str) -> str:
    """arq task. Runs one job and records its outcome, whatever happens."""
    sessionmaker: async_sessionmaker[AsyncSession] = ctx["sessionmaker"]
    storage_root: Path = ctx["storage_root"]

    async with sessionmaker() as db:
        job = await db.get(Job, job_id)
        if job is None:
            _logger.error("job %s not found", job_id)
            return "missing"

        video = await db.get(Video, job.video_id)
        if video is None:
            job.status = JobStatus.FAILED
            job.error_message = "the video for this job no longer exists"
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return "missing-video"

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        await db.commit()

        try:
            outcome = process_video_job(job, video, storage_root=storage_root)
        except Exception:
            # process_video_job catches its own failures; reaching here means something
            # outside it broke. The job must still reach a terminal state, or the user
            # waits on a job nothing will ever finish.
            _logger.exception("job %s: unhandled failure in the consumer", job_id)
            job.status = JobStatus.FAILED
            job.error_message = "processing failed; the file could not be analysed"
            job.finished_at = datetime.now(UTC)
            await db.commit()
            log_security_event(
                SecurityEvent.JOB_FAILED, user_id=job.owner_id, outcome="error", job_id=job_id
            )
            return "failed"

        for record in apply_outcome(job, video, outcome):
            db.add(record)

        await db.commit()

        event = (
            SecurityEvent.JOB_FAILED
            if outcome.status is JobStatus.FAILED
            else SecurityEvent.JOB_SUCCEEDED
        )
        log_security_event(
            event,
            user_id=job.owner_id,
            outcome=outcome.status.value,
            job_id=job_id,
            synthetic=outcome.synthetic,
        )
        return outcome.status.value


async def startup(ctx: dict[str, Any]) -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set; the worker cannot start without it")
    engine = create_async_engine(database_url, future=True)
    ctx["engine"] = engine
    ctx["sessionmaker"] = async_sessionmaker(engine, expire_on_commit=False)
    ctx["storage_root"] = Path(os.environ.get("STORAGE_ROOT", "./storage"))


async def shutdown(ctx: dict[str, Any]) -> None:
    engine = ctx.get("engine")
    if engine is not None:
        await engine.dispose()


class WorkerSettings:
    """arq configuration. Referenced as `arq tayr.worker.main.WorkerSettings`."""

    functions = [process_job]  # noqa: RUF012
    on_startup = startup
    on_shutdown = shutdown
    # One job at a time. Decoding is CPU- and memory-hungry, and the container's caps
    # are sized for a single job; running several would make the memory limit a
    # coin-flip between them.
    max_jobs = 1
    # A job that has not finished in this long is stuck. The limit is above the maximum
    # permitted video duration by a wide margin.
    job_timeout = 1800
