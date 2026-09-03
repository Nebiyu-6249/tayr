"""API settings, from the environment.

Every secret is read here, server-side, and never leaves the process. Nothing in this
module may be serialised into a response or reach the frontend bundle.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./tayr.db"
    redis_url: str = "redis://localhost:6379/0"

    secret_key: str = Field(default="", min_length=0)

    # Explicit allowlist. Never "*", and never a reflection of the Origin header:
    # reflecting Origin with credentials enabled is equivalent to having no CORS at all.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    storage_root: str = "./storage"
    max_upload_bytes: int = 512 * 1024 * 1024

    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_monthly_usd_cap: float = 10.0

    # Slack. The signing secret is what authenticates inbound interactivity requests;
    # without it the callback endpoint refuses every request rather than accepting
    # unverified ones.
    slack_signing_secret: str = ""
    slack_bot_token: str = ""
    slack_alert_channel: str = "#airspace-alerts"
    slack_audit_channel: str = "#airspace-audit"
    site_registry_path: str = "configs/sites/demo.yaml"

    enable_hsts: bool = True
    require_secure_cookies: bool = True

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError(
                "CORS origin '*' is not permitted. With credentials enabled a wildcard "
                "lets any site read authenticated responses. List origins explicitly."
            )
        return value
