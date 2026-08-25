"""arq consumer tests.

The behaviour that matters: a job must ALWAYS reach a terminal state. A job left as
`running` after a crash is indistinguishable from one still working, and the user waits
on something nothing will ever finish.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tayr.db.models import Base, Job, JobStatus, User, Video
from tayr.worker.main import process_job

pytest.importorskip("av", reason="PyAV is in the 'cv' extra")

from tests.test_worker_probe import make_video


async def _setup(tmp_path: Path) -> tuple[dict[str, Any], str]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    storage = tmp_path / "storage"
    storage.mkdir()
    make_video(storage / "clip.mp4")

    async with sessionmaker() as db:
        user = User(id="u1", email="a@example.com", password_hash="x")  # noqa: S106
        video = Video(
            id="v1",
            owner_id="u1",
            storage_name="clip.mp4",
            original_filename="c.mp4",
            container="mp4",
            size_bytes=(storage / "clip.mp4").stat().st_size,
        )
        job = Job(id="j1", owner_id="u1", video_id="v1", status=JobStatus.QUEUED)
        db.add_all([user, video, job])
        await db.commit()

    return {"sessionmaker": sessionmaker, "storage_root": storage}, "j1"


class TestConsumer:
    def test_job_reaches_a_terminal_state(self, tmp_path: Path) -> None:
        async def run() -> str:
            ctx, job_id = await _setup(tmp_path)
            result: str = await process_job(ctx, job_id)
            async with ctx["sessionmaker"]() as db:
                job = await db.get(Job, job_id)
                assert job is not None
                assert job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
                assert job.finished_at is not None
            return result

        assert asyncio.run(run()) == "succeeded"

    def test_probed_dimensions_are_persisted(self, tmp_path: Path) -> None:
        """The API cannot show resolution until the worker has opened the container."""

        async def run() -> None:
            ctx, job_id = await _setup(tmp_path)
            await process_job(ctx, job_id)
            async with ctx["sessionmaker"]() as db:
                video = await db.get(Video, "v1")
                assert video is not None
                assert video.width == 320
                assert video.height == 240
                assert video.fps is not None and video.fps > 0
                assert video.duration_seconds is not None

        asyncio.run(run())

    def test_synthetic_flag_reaches_the_job_row(self, tmp_path: Path) -> None:
        async def run() -> None:
            ctx, job_id = await _setup(tmp_path)
            await process_job(ctx, job_id)
            async with ctx["sessionmaker"]() as db:
                job = await db.get(Job, job_id)
                assert job is not None
                assert job.synthetic is True  # StubDetector is not a real detector

        asyncio.run(run())

    def test_missing_job_is_handled(self, tmp_path: Path) -> None:
        async def run() -> str:
            ctx, _ = await _setup(tmp_path)
            return await process_job(ctx, "does-not-exist")

        assert asyncio.run(run()) == "missing"

    def test_deleting_a_video_cascades_to_its_jobs(self, tmp_path: Path) -> None:
        """Deleting a video removes its jobs, so the consumer finds nothing to run.
        This is the normal path when a user deletes an upload mid-processing."""

        async def run() -> None:
            ctx, job_id = await _setup(tmp_path)
            async with ctx["sessionmaker"]() as db:
                video = await db.get(Video, "v1")
                assert video is not None
                await db.delete(video)
                await db.commit()
                assert await db.get(Job, job_id) is None
            assert await process_job(ctx, job_id) == "missing"

        asyncio.run(run())

    def test_job_with_a_dangling_video_reference_still_terminates(self, tmp_path: Path) -> None:
        """Defensive: a job whose video row is absent without a cascade (database
        surgery, a partial restore) must still reach a terminal state rather than
        leaving the user waiting."""

        async def run() -> None:
            ctx, _ = await _setup(tmp_path)
            async with ctx["sessionmaker"]() as db:
                db.add(
                    Job(
                        id="orphan",
                        owner_id="u1",
                        video_id="no-such-video",
                        status=JobStatus.QUEUED,
                    )
                )
                await db.commit()
            assert await process_job(ctx, "orphan") == "missing-video"
            async with ctx["sessionmaker"]() as db:
                job = await db.get(Job, "orphan")
                assert job is not None
                assert job.status == JobStatus.FAILED
                assert job.finished_at is not None

        asyncio.run(run())

    def test_unhandled_failure_still_writes_a_terminal_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash outside process_video_job must not leave the job running forever."""

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("something unexpected")

        monkeypatch.setattr("tayr.worker.main.process_video_job", boom)

        async def run() -> None:
            ctx, job_id = await _setup(tmp_path)
            assert await process_job(ctx, job_id) == "failed"
            async with ctx["sessionmaker"]() as db:
                job = await db.get(Job, job_id)
                assert job is not None
                assert job.status == JobStatus.FAILED
                assert job.finished_at is not None
                # Generic message: the exception text could carry a path or a query.
                assert "something unexpected" not in (job.error_message or "")

        asyncio.run(run())
