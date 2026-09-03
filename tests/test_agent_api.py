"""Agent API tests.

The Slack callback tests matter most: that endpoint has no session, so its signature
check is the only thing standing between the public internet and the incident records.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tayr.agent.slack import compute_signature
from tayr.api.app import create_app
from tayr.api.auth import CSRF_COOKIE
from tayr.api.settings import ApiSettings
from tayr.db.models import AgentDecisionRecord, Base, Job, User, Video

SECRET = "test-signing-secret-0123456789abcdef"  # noqa: S105 - fixture
PASSWORD = "correct-horse-battery-staple"  # noqa: S105 - fixture


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "agent.db"
    settings = ApiSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=str(tmp_path / "storage"),
        cors_origins=["http://localhost:3000"],
        slack_signing_secret=SECRET,
    )

    async def seed() -> None:
        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as db:
            from tayr.security.passwords import hash_password

            owner = User(
                id="u-owner", email="owner@example.com", password_hash=hash_password(PASSWORD)
            )
            other = User(
                id="u-other", email="other@example.com", password_hash=hash_password(PASSWORD)
            )
            video = Video(
                id="v1",
                owner_id="u-owner",
                storage_name="a.mp4",
                original_filename="a.mp4",
                container="mp4",
                size_bytes=10,
            )
            job = Job(id="j1", owner_id="u-owner", video_id="v1", status="succeeded")
            decision = AgentDecisionRecord(
                id="d1",
                owner_id="u-owner",
                job_id="j1",
                track_id="trk-1",
                site_id="demo-north",
                verdict="escalate",
                attention="immediate",
                uncertainty="no_classifier_trained",
                rule_id="uncertain.no_classifier",
                record_json=json.dumps(
                    {
                        "rule_id": "uncertain.no_classifier",
                        "tool_calls": [{"tool_name": "analyze_track"}],
                    }
                ),
                audit_hash="a" * 64,
                prompt_version="1.0.0",
            )
            db.add_all([owner, other, video, job, decision])
            await db.commit()
        await engine.dispose()

    asyncio.run(seed())
    with TestClient(create_app(settings), base_url="https://testserver") as c:
        yield c


def login(client: TestClient, email: str = "owner@example.com") -> Response:
    return client.post("/auth/login", json={"email": email, "password": PASSWORD})


def csrf(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies.get(CSRF_COOKIE) or ""}


def slack_body(action: str = "confirm", track_id: str = "trk-1") -> str:
    return urlencode(
        {
            "payload": json.dumps(
                {
                    "type": "block_actions",
                    "user": {"id": "U1", "name": "sam"},
                    "actions": [{"action_id": action, "value": track_id, "type": "button"}],
                }
            )
        }
    )


def signed_headers(body: str, *, secret: str = SECRET) -> dict[str, str]:
    ts = str(int(time.time()))
    return {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": compute_signature(signing_secret=secret, timestamp=ts, body=body),
        "Content-Type": "application/x-www-form-urlencoded",
    }


class TestDecisionReads:
    def test_owner_lists_decisions_for_a_job(self, client: TestClient) -> None:
        login(client)
        rows = client.get("/jobs/j1/decisions").json()
        assert len(rows) == 1
        assert rows[0]["verdict"] == "escalate"

    def test_detail_returns_the_full_tool_trace(self, client: TestClient) -> None:
        """An operator who cannot audit a dismissal has no reason to trust one."""
        login(client)
        body = client.get("/decisions/d1").json()
        assert body["record"]["rule_id"] == "uncertain.no_classifier"
        assert body["record"]["tool_calls"][0]["tool_name"] == "analyze_track"

    def test_owner_id_is_not_serialised(self, client: TestClient) -> None:
        login(client)
        assert "owner_id" not in client.get("/decisions/d1").json()["decision"]

    def test_anonymous_is_refused(self, client: TestClient) -> None:
        assert client.get("/decisions/d1").status_code == 401
        assert client.get("/jobs/j1/decisions").status_code == 401

    def test_another_tenant_gets_404_not_403(self, client: TestClient) -> None:
        """403 would confirm the id exists."""
        login(client, "other@example.com")
        real = client.get("/decisions/d1")
        fake = client.get("/decisions/00000000-0000-0000-0000-000000000000")
        assert real.status_code == fake.status_code == 404
        assert real.json() == fake.json()

    def test_another_tenant_cannot_list_a_job(self, client: TestClient) -> None:
        login(client, "other@example.com")
        assert client.get("/jobs/j1/decisions").status_code == 404


class TestWebFeedback:
    def test_owner_records_feedback(self, client: TestClient) -> None:
        login(client)
        r = client.post(
            "/decisions/d1/feedback",
            json={"response": "confirmed", "note": "checked the clip"},
            headers=csrf(client),
        )
        assert r.status_code == 201
        assert client.get("/decisions/d1").json()["feedback"][0]["response"] == "confirmed"

    def test_feedback_is_written_alongside_the_decision_not_over_it(
        self, client: TestClient
    ) -> None:
        """Agreement and disagreement are both signal; overwriting loses one."""
        login(client)
        client.post(
            "/decisions/d1/feedback", json={"response": "dismissed_as_bird"}, headers=csrf(client)
        )
        body = client.get("/decisions/d1").json()
        assert body["decision"]["verdict"] == "escalate"  # unchanged
        assert body["feedback"][0]["response"] == "dismissed_as_bird"

    def test_invalid_response_value_is_rejected(self, client: TestClient) -> None:
        login(client)
        r = client.post(
            "/decisions/d1/feedback", json={"response": "delete_everything"}, headers=csrf(client)
        )
        assert r.status_code == 422

    def test_feedback_requires_csrf(self, client: TestClient) -> None:
        login(client)
        assert (
            client.post("/decisions/d1/feedback", json={"response": "confirmed"}).status_code == 403
        )

    def test_another_tenant_cannot_leave_feedback(self, client: TestClient) -> None:
        login(client, "other@example.com")
        r = client.post(
            "/decisions/d1/feedback", json={"response": "confirmed"}, headers=csrf(client)
        )
        assert r.status_code == 404


class TestSlackCallback:
    """No session here. The signature is the authentication."""

    def test_a_signed_request_records_feedback(self, client: TestClient) -> None:
        body = slack_body()
        r = client.post("/slack/interactions", content=body, headers=signed_headers(body))
        assert r.status_code == 200
        assert "Recorded" in r.json()["text"]

    def test_an_unsigned_request_is_rejected(self, client: TestClient) -> None:
        """THE attack: find the URL, POST to it, mutate incident records."""
        body = slack_body()
        r = client.post(
            "/slack/interactions",
            content=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 401

    def test_a_forged_signature_is_rejected(self, client: TestClient) -> None:
        body = slack_body()
        headers = signed_headers(body, secret="wrong-secret")  # noqa: S106 - fixture
        assert client.post("/slack/interactions", content=body, headers=headers).status_code == 401

    def test_a_tampered_body_is_rejected(self, client: TestClient) -> None:
        """Sign one payload, send another."""
        signed = slack_body("confirm", "trk-1")
        headers = signed_headers(signed)
        tampered = slack_body("mark_authorized", "trk-1")
        assert (
            client.post("/slack/interactions", content=tampered, headers=headers).status_code == 401
        )

    def test_a_replayed_old_request_is_rejected(self, client: TestClient) -> None:
        body = slack_body()
        old = str(int(time.time()) - 3600)
        headers = {
            "X-Slack-Request-Timestamp": old,
            "X-Slack-Signature": compute_signature(signing_secret=SECRET, timestamp=old, body=body),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        assert client.post("/slack/interactions", content=body, headers=headers).status_code == 401

    def test_rejection_leaks_no_detail(self, client: TestClient) -> None:
        """An attacker probing this endpoint should learn nothing about why it failed."""
        body = slack_body()
        r = client.post(
            "/slack/interactions",
            content=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.json()["detail"] == "unauthorized"

    def test_an_unknown_action_is_refused(self, client: TestClient) -> None:
        body = slack_body("delete_everything")
        r = client.post("/slack/interactions", content=body, headers=signed_headers(body))
        assert r.status_code == 400

    def test_an_unknown_track_does_not_leak_existence(self, client: TestClient) -> None:
        body = slack_body("confirm", "no-such-track")
        r = client.post("/slack/interactions", content=body, headers=signed_headers(body))
        assert r.status_code == 200
        assert "no longer available" in r.json()["text"]

    def test_slack_feedback_is_attributed_to_the_decision_owner(self, client: TestClient) -> None:
        """A Slack user is not a Tayr user. The username is display text and grants
        nothing."""
        body = slack_body()
        client.post("/slack/interactions", content=body, headers=signed_headers(body))
        login(client)
        feedback = client.get("/decisions/d1").json()["feedback"]
        assert feedback[0]["responder"] == "sam"
        assert feedback[0]["response"] == "confirmed"

    @pytest.mark.parametrize(
        ("action", "stored"),
        [
            ("confirm", "confirmed"),
            ("dismiss_as_bird", "dismissed_as_bird"),
            ("mark_authorized", "marked_authorized"),
        ],
    )
    def test_all_three_buttons_write_back(
        self, client: TestClient, action: str, stored: str
    ) -> None:
        body = slack_body(action)
        assert (
            client.post(
                "/slack/interactions", content=body, headers=signed_headers(body)
            ).status_code
            == 200
        )
        login(client)
        assert client.get("/decisions/d1").json()["feedback"][0]["response"] == stored
