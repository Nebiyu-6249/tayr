"""Database models.

Two rules shape this schema:

**Every user-owned row carries `owner_id`, and every query filters on it.** Authorisation
is a per-record check on reads and writes, not a filter applied only to list endpoints.
The classic breach is an endpoint that lists correctly but lets
`GET /videos/{someone-elses-id}` through.

**Nothing here is serialised to a client directly.** Responses are built from explicit
schemas in `tayr.api.schemas`, so adding a column can never accidentally publish it.
`password_hash` and `session_token_hash` are the obvious cases, but `storage_path` also
leaks server filesystem layout.

PostgreSQL row-level security is the second layer, applied in the migration rather than
here, so an application bug cannot leak across tenants on its own. See docs/SECURITY.md.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Per-user storage quota. Without one, a single visitor fills the disk.
    storage_quota_bytes: Mapped[int] = mapped_column(
        BigInteger, default=2 * 1024 * 1024 * 1024, nullable=False
    )
    storage_used_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    # Bumped on password change, invalidating every existing session at once.
    session_epoch: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    videos: Mapped[list[Video]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    sessions: Mapped[list[Session]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Session(Base):
    """A login session. The token itself is never stored - only its SHA-256."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    csrf_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Snapshot of the user's session_epoch at issue time; a mismatch invalidates.
    epoch: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped[User] = relationship(back_populates="sessions")


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Generated, never the user's filename. Not serialised to clients: it would leak
    # server filesystem layout.
    storage_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # Kept only to show the user what they uploaded. Never used to build a path, and
    # escaped on output.
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    container: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Populated by the worker after probing. Null until then.
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    owner: Mapped[User] = relationship(back_populates="videos")
    jobs: Mapped[list[Job]] = relationship(back_populates="video", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_videos_owner_created", "owner_id", "created_at"),)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Denormalised from Video deliberately: it lets an ownership check on a job avoid a
    # join, and an authorisation check that needs a join is one someone will skip.
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    video_id: Mapped[str] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED, nullable=False)
    frames_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Shown to the user. Must never carry a stack trace or internal path - the worker
    # writes a sanitised message here and logs detail server-side.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # True when the pipeline ran on placeholder data. Propagates into the UI and any
    # generated report so a synthetic result is never mistaken for a real one.
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    video: Mapped[Video] = relationship(back_populates="jobs")
    tracks: Mapped[list[TrackRecord]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    @property
    def progress(self) -> float:
        """Real progress in [0, 1], or 0 when the frame count is not yet known.

        Never a fabricated percentage: an unknown total reports 0 and the UI shows an
        indeterminate state rather than a fake bar.
        """
        if not self.frames_total:
            return 0.0
        return min(1.0, self.frames_processed / self.frames_total)


class TrackRecord(Base):
    """One track produced by a job, with the motion features that classified it."""

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    track_number: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    first_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    last_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    median_pixels_on_target: Mapped[float] = mapped_column(Float, nullable=False)
    # The motion features behind the decision, so a result is explainable rather than
    # an opaque label.
    features_json: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped[Job] = relationship(back_populates="tracks")


class AgentDecisionRecord(Base):
    """One agent triage decision. Written once, never updated.

    Immutability is the audit property. Operator feedback lands in
    `OperatorFeedbackRecord`, keyed to this row, rather than mutating it - otherwise
    "what did the agent decide" and "what did the human decide" become the same field
    and the disagreement, which is the interesting signal, is lost.
    """

    __tablename__ = "agent_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    site_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    attention: Mapped[str] = mapped_column(String(16), nullable=False)
    uncertainty: Mapped[str] = mapped_column(String(32), nullable=False)
    rule_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # The full record as JSON: every tool call with arguments and results, the
    # rationale bullets, and the model's prose. This is what a human reconstructs the
    # decision from.
    record_json: Mapped[str] = mapped_column(Text, nullable=False)
    audit_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    prose_diverged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True when the model's stated verdict disagreed with the computed one. A defect
    signal, surfaced rather than swallowed."""

    synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    model: Mapped[str] = mapped_column(String(64), default="none", nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rounds_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    round_cap_reached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Set when an escalation has been posted, so a re-run updates rather than reposts.
    notification_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)

    feedback: Mapped[list[OperatorFeedbackRecord]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # One decision per track. The agent is idempotent per track by construction,
        # not by convention - a duplicate insert fails rather than double-paging.
        UniqueConstraint("track_id", name="uq_agent_decisions_track"),
        Index("ix_agent_decisions_site_created", "site_id", "created_at"),
    )


class OperatorFeedbackRecord(Base):
    """A human's response to an agent decision.

    This is the feedback loop, and later the labelled data: every button press is a
    human-confirmed label on a track whose motion features are already computed. It is
    stored separately from the decision so that agreement and disagreement are both
    recoverable.
    """

    __tablename__ = "operator_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("agent_decisions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # confirmed | dismissed_as_bird | marked_authorized
    response: Mapped[str] = mapped_column(String(32), nullable=False)
    # Who pressed it, as reported by the notification surface. Untrusted display text:
    # escaped on output, never used for authorisation.
    responder: Mapped[str] = mapped_column(String(128), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    decision: Mapped[AgentDecisionRecord] = relationship(back_populates="feedback")
