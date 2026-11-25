"""Alembic environment — offline/online modes.

The migration chain manages the relational tables; PostGIS-specific DDL
(CREATE EXTENSION, geometry columns, GiST indexes) lives in db/schema.sql and
is applied by the 0001 migration via raw SQL guarded for Postgres.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# DATABASE_URL (docker compose) overrides the ini file
import os

url = os.environ.get("DATABASE_URL")
if url:
    # alembic expects the sqlalchemy driver form
    config.set_main_option(
        "sqlalchemy.url",
        url.replace("postgresql://", "postgresql+psycopg2://"),
    )

target_metadata = None  # migrations are explicit; no autogenerate metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
