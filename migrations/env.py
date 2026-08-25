"""Alembic environment.

The database URL comes from DATABASE_URL, never from alembic.ini, so no connection
string is ever committed.

Migrations run as a **more privileged role** than the application. The app role has no
DDL rights, which is what stops a SQL-injection bug from rewriting the schema or
disabling row-level security. See docs/SECURITY.md.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from tayr.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Migrations take the URL from the environment so "
            "that no connection string is committed to the repository."
        )
    # Alembic runs synchronously; strip an async driver if one is configured.
    return url.replace("+asyncpg", "").replace("+aiosqlite", "").replace("+psycopg", "+psycopg")


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
