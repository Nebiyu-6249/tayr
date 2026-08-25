"""Video upload and job endpoints.

Every read and write here goes through `require_owned`, which returns 404 rather than
403 for someone else's record. The upload path validates by content, enforces a quota,
and never touches the user's filename.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy import select

from tayr.api.auth import require_owned
from tayr.api.deps import CsrfProtected, CurrentUser, DbSession, Settings
from tayr.api.schemas import JobOut, JobResultOut, TrackOut, VideoOut
from tayr.db.models import Job, JobStatus, TrackRecord, Video
from tayr.security.uploads import (
    UploadRejectedError,
    VideoLimits,
    generate_storage_name,
    is_safe_storage_path,
    validate_upload,
)

router = APIRouter(tags=["videos"])

# Read in bounded chunks so the size cap is enforced *during* streaming. Buffering the
# whole body first and checking afterwards means the exhaustion already happened, and
# the Content-Length header is attacker-controlled anyway.
_CHUNK = 1024 * 1024
_HEADER_BYTES = 32


@router.post("/videos", response_model=VideoOut, status_code=status.HTTP_201_CREATED)
async def upload_video(
    db: DbSession,
    user: CurrentUser,
    settings: Settings,
    _: CsrfProtected,
    file: Annotated[UploadFile, File()],
) -> Video:
    """Accept a video upload."""
    limits = VideoLimits(max_bytes=settings.max_upload_bytes)

    remaining_quota = user.storage_quota_bytes - user.storage_used_bytes
    if remaining_quota <= 0:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "storage quota exhausted; delete a video to free space",
        )

    storage_root = Path(settings.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    header = await file.read(_HEADER_BYTES)
    try:
        container = validate_upload(header, max(len(header), 1), limits=limits)
    except UploadRejectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    storage_name = generate_storage_name(container)
    destination = storage_root / storage_name
    if not is_safe_storage_path(destination, root=storage_root):
        # Unreachable with a generated name; kept as a tripwire for future changes.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid storage path")

    size = 0
    try:
        with destination.open("wb") as out:
            out.write(header)
            size += len(header)
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if size > limits.max_bytes or size > remaining_quota:
                    raise UploadRejectedError(
                        "upload exceeds the size limit or your remaining storage quota"
                    )
                out.write(chunk)
    except UploadRejectedError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    video = Video(
        owner_id=user.id,
        storage_name=storage_name,
        original_filename=Path(file.filename or "upload").name[:255],
        container=container,
        size_bytes=size,
    )
    user.storage_used_bytes += size
    db.add(video)
    await db.flush()
    return video


@router.get("/videos", response_model=list[VideoOut])
async def list_videos(db: DbSession, user: CurrentUser) -> list[Video]:
    result = await db.execute(
        select(Video).where(Video.owner_id == user.id).order_by(Video.created_at.desc())
    )
    return list(result.scalars().all())


@router.get("/videos/{video_id}", response_model=VideoOut)
async def get_video(video_id: str, db: DbSession, user: CurrentUser) -> Video:
    return await require_owned(db, Video, video_id, user)


@router.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: str, db: DbSession, user: CurrentUser, settings: Settings, _: CsrfProtected
) -> None:
    video = await require_owned(db, Video, video_id, user)
    path = Path(settings.storage_root) / video.storage_name
    if is_safe_storage_path(path, root=Path(settings.storage_root)):
        path.unlink(missing_ok=True)
    user.storage_used_bytes = max(0, user.storage_used_bytes - video.size_bytes)
    await db.delete(video)


@router.post("/videos/{video_id}/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED)
async def create_job(video_id: str, db: DbSession, user: CurrentUser, _: CsrfProtected) -> Job:
    """Queue processing for a video. Returns immediately; never blocks on decode."""
    video = await require_owned(db, Video, video_id, user)
    job = Job(owner_id=user.id, video_id=video.id, status=JobStatus.QUEUED)
    db.add(job)
    await db.flush()
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str, db: DbSession, user: CurrentUser) -> Job:
    return await require_owned(db, Job, job_id, user)


@router.get("/jobs/{job_id}/results", response_model=JobResultOut)
async def get_job_results(job_id: str, db: DbSession, user: CurrentUser) -> JobResultOut:
    job = await require_owned(db, Job, job_id, user)
    result = await db.execute(
        select(TrackRecord).where(TrackRecord.job_id == job.id).order_by(TrackRecord.track_number)
    )
    tracks = [
        TrackOut(
            id=t.id,
            track_number=t.track_number,
            label=t.label,
            confidence=t.confidence,
            first_frame=t.first_frame,
            last_frame=t.last_frame,
            median_pixels_on_target=t.median_pixels_on_target,
            features=json.loads(t.features_json),
        )
        for t in result.scalars().all()
    ]
    return JobResultOut(job=JobOut.model_validate(job), tracks=tracks, synthetic=job.synthetic)
