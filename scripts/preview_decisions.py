"""Seed a local SQLite database from a real `decisions.json`, for the web UI.

## Why this exists

`docker compose up` has never completed end to end here - Docker Hub rate-limits
anonymous pulls through this environment's proxy - so the frontend has never been seen
rendering a decision that came out of a real run. That is a gap in what can honestly be
claimed about Phase 8, and it does not need Docker to close: the API runs against SQLite,
the frontend is a static Next.js dev server, and neither needs Postgres, Redis, or a
worker to answer `GET /decisions/{id}`.

This writes real `AgentDecision` records - the same rows the worker would write, through
the same models - so what the page renders is the run's own output rather than a fixture
written to look like one.

## Not for anything but a local preview

It creates a user and prints its password. The password is generated with `secrets` and
is different every time, the database is a file you name, and `REQUIRE_SECURE_COOKIES`
has to be turned off for the browser to keep a session over plain http. All three are
reasons this is a desk tool and never a deployment step.

Usage:

    python scripts/preview_decisions.py --decisions watch-out/decisions.json

It prints the login, the decision URLs, and the two commands to run next.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tayr.agent.records import audit_hash_of  # noqa: E402
from tayr.db.models import (  # noqa: E402
    AgentDecisionRecord,
    Base,
    Job,
    TrackRecord,
    User,
    Video,
)
from tayr.security.passwords import hash_password  # noqa: E402

OWNER_ID = "preview-owner"
VIDEO_ID = "preview-video"
JOB_ID = "preview-job"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        required=True,
        help="decisions.json from `tayr watch run` or `tayr watch demo`.",
    )
    parser.add_argument(
        "--db", type=Path, default=Path("preview.db"), help="SQLite file to create."
    )
    # A real domain shape, because the API validates with pydantic's EmailStr and
    # "preview@localhost" is rejected for having no period after the @.
    parser.add_argument("--email", default="preview@example.com", help="Login to create.")
    return parser.parse_args()


def track_number(track_id: str) -> str:
    """The tracker id embedded in a decision's track_id, or the whole thing."""
    return track_id.rsplit("-t", 1)[-1] or track_id


async def seed(records: list[dict[str, Any]], *, db_path: Path, email: str, password: str) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    if db_path.exists():
        # Refuse rather than merge: a half-overwritten preview database is a confusing
        # thing to debug, and the fix is one `rm`.
        raise SystemExit(f"{db_path} already exists. Remove it first.")

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(User(id=OWNER_ID, email=email, password_hash=hash_password(password)))
        session.add(
            Video(
                id=VIDEO_ID,
                owner_id=OWNER_ID,
                storage_name="preview.mp4",
                original_filename="preview.mp4",
                container="mp4",
                size_bytes=0,
            )
        )
        synthetic = any(bool(r.get("synthetic")) for r in records)
        session.add(
            Job(
                id=JOB_ID,
                owner_id=OWNER_ID,
                video_id=VIDEO_ID,
                status="succeeded",
                synthetic=synthetic,
            )
        )

        for index, record in enumerate(records):
            track_id = f"trk-{index}"
            session.add(
                TrackRecord(
                    id=track_id,
                    job_id=JOB_ID,
                    owner_id=OWNER_ID,
                    track_number=int(track_number(record["track_id"]) or index),
                    # The classifier is not trained, and this row must not imply it is.
                    label="unknown",
                    confidence=0.0,
                    first_frame=0,
                    last_frame=0,
                    median_pixels_on_target=0.0,
                    features_json="{}",
                )
            )
            session.add(
                AgentDecisionRecord(
                    id=f"d-{index}",
                    owner_id=OWNER_ID,
                    job_id=JOB_ID,
                    track_id=track_id,
                    site_id=record["site_id"],
                    verdict=record["verdict"],
                    attention=record["attention"],
                    uncertainty=record["uncertainty"],
                    rule_id=record["rule_id"],
                    record_json=json.dumps(record),
                    # Recomputed rather than read: `AgentDecision.to_dict()` does not
                    # carry the hash, so a `.get()` here would store an empty string and
                    # the page would show a blank audit field as though one existed.
                    audit_hash=audit_hash_of(record),
                    prose_diverged=bool(record.get("prose_diverged")),
                    # Carried from the run, never assumed. A preview of a synthetic run
                    # must still say it was synthetic.
                    synthetic=bool(record.get("synthetic")),
                    model=record.get("model", "none"),
                    prompt_version=record.get("prompt_version", "unknown"),
                    rounds_used=int(record.get("rounds_used", 0)),
                    round_cap_reached=bool(record.get("round_cap_reached")),
                )
            )
        await session.commit()
    await engine.dispose()


def main() -> None:
    args = parse_args()
    records = json.loads(args.decisions.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{args.decisions} holds no decisions to preview.")

    # `secrets`, not `random`: CLAUDE.md section 3. A predictable password on a local
    # database is still a predictable password, and this one is printed to a terminal.
    password = secrets.token_urlsafe(18)
    asyncio.run(seed(records, db_path=args.db, email=args.email, password=password))

    print(f"seeded {len(records)} decision(s) into {args.db}")
    print(f"  login    {args.email}")
    print(f"  password {password}")
    print("")
    print("Then, in two terminals:")
    print(
        f"  DATABASE_URL=sqlite+aiosqlite:///{args.db} REQUIRE_SECURE_COOKIES=false "
        "ENABLE_HSTS=false uvicorn tayr.api.app:create_app --factory --port 8000"
    )
    print("  cd frontend && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev")
    print("")
    print("Log in at http://localhost:3000/login, then open:")
    for index, record in enumerate(records):
        print(f"  http://localhost:3000/decisions/d-{index}   {record['rule_id']}")


if __name__ == "__main__":
    main()
