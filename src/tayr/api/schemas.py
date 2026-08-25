"""Response and request schemas.

Responses are built from these explicitly rather than by serialising an ORM object.
That is the control that stops a future column - a password hash, an internal storage
path, another user's id - from being published by accident. Adding a field to a model
does nothing until it is added here on purpose.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    password: str = Field(min_length=12, max_length=1024)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    password: str = Field(max_length=1024)


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(max_length=1024)
    new_password: str = Field(min_length=12, max_length=1024)


class UserOut(_Out):
    """Note what is absent: password_hash, session_epoch, is_active."""

    id: str
    email: str
    created_at: datetime
    storage_quota_bytes: int
    storage_used_bytes: int


class VideoOut(_Out):
    """Note what is absent: storage_name and owner_id.

    storage_name would leak server filesystem layout; owner_id tells a caller nothing
    they do not already know and would confirm the existence of other users' ids.
    """

    id: str
    original_filename: str
    container: str
    size_bytes: int
    created_at: datetime
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None


class JobOut(_Out):
    id: str
    video_id: str
    status: str
    progress: float
    frames_processed: int
    frames_total: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    synthetic: bool = False


class TrackOut(_Out):
    id: str
    track_number: int
    label: str
    confidence: float
    first_frame: int
    last_frame: int
    median_pixels_on_target: float
    features: dict[str, float]


class JobResultOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job: JobOut
    tracks: list[TrackOut]
    synthetic: bool = False
    """Mirrors job.synthetic at the top level so a client cannot render results
    without having seen the flag."""


class ErrorOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str
