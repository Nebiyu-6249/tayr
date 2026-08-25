"""row level security

Second layer beneath the application's per-record ownership checks. If an endpoint ever
forgets `require_owned`, or a SQL-injection bug reaches the database directly, these
policies still confine the connection to one tenant's rows.

The app connects as `tayr_app`, which:
  - has no DDL rights, so injected SQL cannot alter the schema or drop the policies
  - is NOT the table owner, so it cannot bypass RLS (an owner is exempt by default)
  - has NOBYPASSRLS explicitly

Every request sets `tayr.current_user_id` for its transaction; the policies compare
each row's owner_id against it. A request that sets nothing sees nothing, which is the
safe default.

PostgreSQL only. SQLite has no RLS, so the test suite exercises the application-level
checks and this layer is verified against Postgres in deployment. That gap is recorded
in docs/THREAT_MODEL.md.

Revision ID: b1a2c3d4e5f6
Revises: fa384ae4a084
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b1a2c3d4e5f6"
down_revision: str | None = "fa384ae4a084"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OWNED_TABLES = ("videos", "jobs", "tracks")


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        # SQLite and others have no RLS. Skipping keeps the migration runnable in tests
        # while the real protection applies wherever it can.
        return

    for table in _OWNED_TABLES:
        op.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        # FORCE makes the policy apply to the table owner too, so a mistake in role
        # assignment does not silently disable it.
        op.execute(sa.text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        op.execute(
            sa.text(
                f"""
                CREATE POLICY {table}_owner_isolation ON {table}
                USING (owner_id = current_setting('tayr.current_user_id', true))
                WITH CHECK (owner_id = current_setting('tayr.current_user_id', true))
                """
            )
        )

    # A user may only read their own row.
    op.execute(sa.text("ALTER TABLE users ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("ALTER TABLE users FORCE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            """
            CREATE POLICY users_self_isolation ON users
            USING (id = current_setting('tayr.current_user_id', true))
            WITH CHECK (id = current_setting('tayr.current_user_id', true))
            """
        )
    )

    # Sessions are looked up by token hash before the user is known, so they cannot be
    # gated on current_user_id. They are protected instead by the token being a 256-bit
    # secret stored only as a hash. Recorded as an accepted risk in docs/THREAT_MODEL.md.


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(sa.text("DROP POLICY IF EXISTS users_self_isolation ON users"))
    op.execute(sa.text("ALTER TABLE users DISABLE ROW LEVEL SECURITY"))
    for table in _OWNED_TABLES:
        op.execute(sa.text(f"DROP POLICY IF EXISTS {table}_owner_isolation ON {table}"))
        op.execute(sa.text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))
