"""API tests, end-to-end against a real SQLite database.

The security tests here matter more than the happy paths. Each one names the attack it
prevents, so a future change that breaks a control fails with an explanation rather than
just a red assertion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy.ext.asyncio import create_async_engine

from tayr.api.app import create_app
from tayr.api.auth import CSRF_COOKIE, SESSION_COOKIE
from tayr.api.settings import ApiSettings
from tayr.db.models import Base

MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 512
PASSWORD = "correct-horse-battery-staple"  # noqa: S105 - test fixture, not a credential
OTHER_PASSWORD = "another-perfectly-fine-passphrase"  # noqa: S105


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "test.db"
    settings = ApiSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=str(tmp_path / "storage"),
        cors_origins=["http://localhost:3000"],
        max_upload_bytes=10 * 1024 * 1024,
    )

    async def _create() -> None:
        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    app = create_app(settings)
    # The __Host- cookie prefix requires Secure, so the test transport must be https.
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def register(client: TestClient, email: str, password: str = PASSWORD) -> Response:
    return client.post("/auth/register", json={"email": email, "password": password})


def login(client: TestClient, email: str, password: str = PASSWORD) -> Response:
    return client.post("/auth/login", json={"email": email, "password": password})


def csrf(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies.get(CSRF_COOKIE) or ""}


class TestHealthAndHeaders:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_security_headers_present(self, client: TestClient) -> None:
        h = client.get("/health").headers
        assert "default-src 'self'" in h["content-security-policy"]
        assert h["x-frame-options"] == "DENY"
        assert h["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in h["content-security-policy"]

    def test_csp_forbids_inline_script(self, client: TestClient) -> None:
        """unsafe-inline would make an injected <script> executable again."""
        csp = client.get("/health").headers["content-security-policy"]
        assert "unsafe-inline" not in csp
        assert "unsafe-eval" not in csp

    def test_hsts_on_https(self, client: TestClient) -> None:
        assert "strict-transport-security" in client.get("/health").headers

    def test_openapi_docs_are_disabled(self, client: TestClient) -> None:
        """Interactive docs enumerate every endpoint and schema for whoever finds them."""
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404


class TestRegistrationEnumeration:
    def test_register_succeeds(self, client: TestClient) -> None:
        assert register(client, "a@example.com").status_code == 201

    def test_duplicate_registration_is_indistinguishable(self, client: TestClient) -> None:
        """THE attack: if a duplicate returns 409 while a new address returns 201, the
        endpoint is an account-existence oracle and rate limiting does not fix it."""
        first = register(client, "dup@example.com")
        second = register(client, "dup@example.com")
        assert first.status_code == second.status_code == 201
        assert first.json() == second.json()

    def test_registration_returns_no_user_record(self, client: TestClient) -> None:
        """Returning the existing user's record on a duplicate would be worse than
        merely confirming the address exists."""
        body = register(client, "b@example.com").json()
        assert "email" not in body
        assert "id" not in body

    def test_registration_issues_no_session(self, client: TestClient) -> None:
        r = register(client, "c@example.com")
        assert SESSION_COOKIE not in r.cookies
        assert client.get("/auth/me").status_code == 401

    def test_short_password_rejected(self, client: TestClient) -> None:
        assert register(client, "d@example.com", "short").status_code == 422


class TestLogin:
    def test_login_after_register(self, client: TestClient) -> None:
        register(client, "e@example.com")
        r = login(client, "e@example.com")
        assert r.status_code == 200
        assert r.json()["email"] == "e@example.com"

    def test_login_sets_hardened_cookies(self, client: TestClient) -> None:
        register(client, "f@example.com")
        r = login(client, "f@example.com")
        raw = r.headers.get_list("set-cookie")
        session_cookie = next(c for c in raw if c.startswith(SESSION_COOKIE))
        assert "HttpOnly" in session_cookie  # XSS cannot read it
        assert "Secure" in session_cookie
        assert "samesite=lax" in session_cookie.lower()
        assert "Path=/" in session_cookie
        # __Host- prefix means the browser rejects it if a Domain attribute is set,
        # which is what stops a subdomain from planting a session cookie.
        assert "Domain=" not in session_cookie

    def test_unknown_user_and_wrong_password_are_identical(self, client: TestClient) -> None:
        """Different messages or status codes here make login an enumeration oracle."""
        register(client, "g@example.com")
        wrong = login(client, "g@example.com", "definitely-not-the-password")
        missing = login(client, "nobody@example.com", "definitely-not-the-password")
        assert wrong.status_code == missing.status_code == 401
        assert wrong.json() == missing.json()

    def test_password_hash_never_leaves_the_server(self, client: TestClient) -> None:
        register(client, "h@example.com")
        body = login(client, "h@example.com").json()
        assert "password_hash" not in body
        assert "password" not in body
        assert "session_epoch" not in body

    def test_repeated_failures_eventually_lock_out(self, client: TestClient) -> None:
        register(client, "i@example.com")
        codes = [
            login(client, "i@example.com", "wrong-password-here").status_code for _ in range(8)
        ]
        assert 429 in codes, f"no lockout after 8 failures: {codes}"


class TestAuthenticationRequired:
    @pytest.mark.parametrize(
        ("method", "path"),
        [("get", "/auth/me"), ("get", "/videos"), ("get", "/videos/xyz"), ("get", "/jobs/xyz")],
    )
    def test_protected_endpoints_reject_anonymous(
        self, client: TestClient, method: str, path: str
    ) -> None:
        assert getattr(client, method)(path).status_code == 401

    def test_forged_session_cookie_rejected(self, client: TestClient) -> None:
        client.cookies.set(SESSION_COOKIE, "made-up-token-value")
        assert client.get("/auth/me").status_code == 401


class TestCsrf:
    def test_state_change_without_csrf_header_is_refused(self, client: TestClient) -> None:
        """SameSite=Lax is a browser behaviour; the double-submit token is what does not
        depend on the client honouring it."""
        register(client, "j@example.com")
        login(client, "j@example.com")
        r = client.post("/auth/logout")
        assert r.status_code == 403

    def test_state_change_with_csrf_header_succeeds(self, client: TestClient) -> None:
        register(client, "k@example.com")
        login(client, "k@example.com")
        assert client.post("/auth/logout", headers=csrf(client)).status_code == 204

    def test_wrong_csrf_token_refused(self, client: TestClient) -> None:
        register(client, "l@example.com")
        login(client, "l@example.com")
        r = client.post("/auth/logout", headers={"X-CSRF-Token": "not-the-right-token"})
        assert r.status_code == 403

    def test_safe_methods_need_no_csrf(self, client: TestClient) -> None:
        register(client, "m@example.com")
        login(client, "m@example.com")
        assert client.get("/auth/me").status_code == 200


class TestSessionLifecycle:
    def test_logout_invalidates_the_session(self, client: TestClient) -> None:
        register(client, "n@example.com")
        login(client, "n@example.com")
        client.post("/auth/logout", headers=csrf(client))
        assert client.get("/auth/me").status_code == 401

    def test_password_change_invalidates_every_session(self, client: TestClient) -> None:
        """If a stolen session survived a password change, changing the password would
        not actually be a remedy."""
        register(client, "o@example.com")
        login(client, "o@example.com")
        stolen = client.cookies.get(SESSION_COOKIE)

        r = client.post(
            "/auth/change-password",
            json={"current_password": PASSWORD, "new_password": OTHER_PASSWORD},
            headers=csrf(client),
        )
        assert r.status_code == 204

        client.cookies.clear()
        client.cookies.set(SESSION_COOKIE, stolen or "")
        assert client.get("/auth/me").status_code == 401

    def test_password_change_requires_the_current_password(self, client: TestClient) -> None:
        register(client, "p@example.com")
        login(client, "p@example.com")
        r = client.post(
            "/auth/change-password",
            json={"current_password": "not-the-current-one", "new_password": OTHER_PASSWORD},
            headers=csrf(client),
        )
        assert r.status_code == 401


class TestUploads:
    def _login(self, client: TestClient, email: str = "u@example.com") -> None:
        register(client, email)
        login(client, email)

    def test_upload_and_list(self, client: TestClient) -> None:
        self._login(client)
        r = client.post(
            "/videos", files={"file": ("clip.mp4", MP4, "video/mp4")}, headers=csrf(client)
        )
        assert r.status_code == 201, r.text
        assert r.json()["container"] == "mp4"
        assert len(client.get("/videos").json()) == 1

    def test_storage_name_is_not_returned(self, client: TestClient) -> None:
        """It would leak server filesystem layout."""
        self._login(client)
        body = client.post(
            "/videos", files={"file": ("clip.mp4", MP4, "video/mp4")}, headers=csrf(client)
        ).json()
        assert "storage_name" not in body
        assert "owner_id" not in body

    def test_content_is_checked_not_the_extension(self, client: TestClient) -> None:
        """A shell script named .mp4 with video/mp4 declared must still be rejected."""
        self._login(client)
        r = client.post(
            "/videos",
            files={"file": ("evil.mp4", b"#!/bin/sh\nrm -rf /\n" + b"\x00" * 64, "video/mp4")},
            headers=csrf(client),
        )
        assert r.status_code == 400
        assert "container" in r.json()["detail"]

    def test_oversize_upload_rejected(self, client: TestClient) -> None:
        self._login(client)
        big = MP4 + b"\x00" * (11 * 1024 * 1024)
        r = client.post(
            "/videos", files={"file": ("big.mp4", big, "video/mp4")}, headers=csrf(client)
        )
        assert r.status_code == 413

    def test_rejected_upload_leaves_no_file_behind(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        self._login(client)
        client.post(
            "/videos",
            files={"file": ("big.mp4", MP4 + b"\x00" * (11 * 1024 * 1024), "video/mp4")},
            headers=csrf(client),
        )
        storage = tmp_path / "storage"
        assert not storage.exists() or list(storage.iterdir()) == []

    def test_upload_requires_csrf(self, client: TestClient) -> None:
        self._login(client)
        assert (
            client.post("/videos", files={"file": ("c.mp4", MP4, "video/mp4")}).status_code == 403
        )

    def test_quota_is_tracked(self, client: TestClient) -> None:
        self._login(client)
        before = client.get("/auth/me").json()["storage_used_bytes"]
        client.post("/videos", files={"file": ("c.mp4", MP4, "video/mp4")}, headers=csrf(client))
        assert client.get("/auth/me").json()["storage_used_bytes"] > before

    def test_delete_frees_quota(self, client: TestClient) -> None:
        self._login(client)
        vid = client.post(
            "/videos", files={"file": ("c.mp4", MP4, "video/mp4")}, headers=csrf(client)
        ).json()["id"]
        used = client.get("/auth/me").json()["storage_used_bytes"]
        assert client.delete(f"/videos/{vid}", headers=csrf(client)).status_code == 204
        assert client.get("/auth/me").json()["storage_used_bytes"] < used


class TestTenantIsolation:
    """The breach that matters most in a multi-user demo."""

    def _two_users(self, client: TestClient) -> tuple[str, str]:
        register(client, "alice@example.com")
        login(client, "alice@example.com")
        alice_video = client.post(
            "/videos", files={"file": ("a.mp4", MP4, "video/mp4")}, headers=csrf(client)
        ).json()["id"]
        alice_job = client.post(f"/videos/{alice_video}/jobs", headers=csrf(client)).json()["id"]
        client.post("/auth/logout", headers=csrf(client))
        client.cookies.clear()

        register(client, "bob@example.com")
        login(client, "bob@example.com")
        return alice_video, alice_job

    def test_cannot_read_another_users_video(self, client: TestClient) -> None:
        alice_video, _ = self._two_users(client)
        assert client.get(f"/videos/{alice_video}").status_code == 404

    def test_cannot_read_another_users_job(self, client: TestClient) -> None:
        _, alice_job = self._two_users(client)
        assert client.get(f"/jobs/{alice_job}").status_code == 404
        assert client.get(f"/jobs/{alice_job}/results").status_code == 404

    def test_cannot_delete_another_users_video(self, client: TestClient) -> None:
        alice_video, _ = self._two_users(client)
        assert client.delete(f"/videos/{alice_video}", headers=csrf(client)).status_code == 404

    def test_cannot_queue_a_job_on_another_users_video(self, client: TestClient) -> None:
        alice_video, _ = self._two_users(client)
        assert client.post(f"/videos/{alice_video}/jobs", headers=csrf(client)).status_code == 404

    def test_list_shows_only_own_videos(self, client: TestClient) -> None:
        self._two_users(client)
        assert client.get("/videos").json() == []

    def test_other_users_records_are_404_not_403(self, client: TestClient) -> None:
        """403 would confirm the id exists, turning enumeration into a census of other
        users' data. A caller must not be able to tell 'not yours' from 'no such thing'."""
        alice_video, _ = self._two_users(client)
        real = client.get(f"/videos/{alice_video}")
        fake = client.get("/videos/00000000-0000-0000-0000-000000000000")
        assert real.status_code == fake.status_code == 404
        assert real.json() == fake.json()


class TestJobs:
    def test_job_queued_not_executed_inline(self, client: TestClient) -> None:
        """Submitting must return immediately; video processing never blocks a request."""
        register(client, "q@example.com")
        login(client, "q@example.com")
        vid = client.post(
            "/videos", files={"file": ("c.mp4", MP4, "video/mp4")}, headers=csrf(client)
        ).json()["id"]
        job = client.post(f"/videos/{vid}/jobs", headers=csrf(client)).json()
        assert job["status"] == "queued"
        assert job["progress"] == 0.0

    def test_progress_is_real_not_fabricated(self, client: TestClient) -> None:
        """An unknown frame total reports 0, so the UI shows an indeterminate state
        rather than an invented percentage."""
        register(client, "r@example.com")
        login(client, "r@example.com")
        vid = client.post(
            "/videos", files={"file": ("c.mp4", MP4, "video/mp4")}, headers=csrf(client)
        ).json()["id"]
        job = client.post(f"/videos/{vid}/jobs", headers=csrf(client)).json()
        assert job["frames_total"] is None
        assert job["progress"] == 0.0

    def test_results_carry_the_synthetic_flag(self, client: TestClient) -> None:
        register(client, "s@example.com")
        login(client, "s@example.com")
        vid = client.post(
            "/videos", files={"file": ("c.mp4", MP4, "video/mp4")}, headers=csrf(client)
        ).json()["id"]
        job_id = client.post(f"/videos/{vid}/jobs", headers=csrf(client)).json()["id"]
        body = client.get(f"/jobs/{job_id}/results").json()
        assert body["synthetic"] is False
        assert body["tracks"] == []


class TestCorsPolicy:
    def test_wildcard_origin_is_refused_at_config_time(self) -> None:
        """With credentials enabled, '*' lets any site read authenticated responses."""
        with pytest.raises(ValueError, match="not permitted"):
            ApiSettings(cors_origins=["*"])

    def test_unlisted_origin_gets_no_cors_grant(self, client: TestClient) -> None:
        r = client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"

    def test_listed_origin_is_allowed(self, client: TestClient) -> None:
        r = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
