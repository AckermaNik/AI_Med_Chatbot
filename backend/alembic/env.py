"""Alembic's migration environment for this project.

Alembic executes this file whenever a migration command runs. The important
connection is ``target_metadata = Base.metadata``: importing ``app.models``
registers every SQLAlchemy model, and Alembic then compares that metadata with
the real database when autogenerating a migration.
"""

import asyncio  # Runs the asynchronous database migration function.
import sys  # Lets us adjust Python's module search path below.
from logging.config import fileConfig  # Loads logging settings from alembic.ini.
from pathlib import Path  # Builds a reliable path to the backend directory.

from alembic import context  # Gives this file access to Alembic's migration context.
from sqlalchemy import pool  # Provides the no-pooling option for migrations.
from sqlalchemy.engine import Connection  # Type hint for the synchronous bridge.
from sqlalchemy.ext.asyncio import async_engine_from_config  # Builds an async engine.

# ``env.py`` lives in backend/alembic, while ``app`` lives in backend/app.
# Adding backend to sys.path makes ``from app...`` imports work from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402  # Loads the database URL from settings.
from app.models import Base  # noqa: E402  # Imports models and fills Base.metadata.

config = context.config  # Reads Alembic's configuration object from alembic.ini.

# Keep the database URL in the application's settings/.env, not in alembic.ini.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Apply the logging configuration if Alembic was given a config file.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the schema description Alembic compares with the real database.
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Tell autogenerate which database objects should be considered."""

    # Alembic owns this bookkeeping table, so our models should not manage it.
    if type_ == "table" and name == "alembic_version":
        return False

    # Consider all other tables, columns, indexes, and constraints.
    return True


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""

    # Configure Alembic using only the URL, useful for producing SQL scripts.
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),  # Database connection URL.
        target_metadata=target_metadata,  # The model schema to compare/use.
        literal_binds=True,  # Put values directly into generated SQL.
        dialect_opts={"paramstyle": "named"},  # Use named SQL parameters.
        compare_type=True,  # Detect changes such as String(100) -> String(200).
        include_object=include_object,  # Apply our object filtering rule.
    )

    # Group the generated migration operations into one transaction block.
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure and execute migrations using an existing database connection."""

    # Use the connection supplied by the async-to-sync bridge below.
    context.configure(
        connection=connection,  # The live PostgreSQL connection.
        target_metadata=target_metadata,  # The model schema for autogeneration.
        compare_type=True,  # Compare SQL column types as well as names.
        include_object=include_object,  # Apply our object filtering rule.
    )

    # Group the migration operations into one transaction block.
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Open an async connection, run migrations, then close it."""

    # Build an async SQLAlchemy engine from the URL configured above.
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),  # Read Alembic settings.
        prefix="sqlalchemy.",  # Use sqlalchemy.* settings from the config.
        poolclass=pool.NullPool,  # Do not keep migration connections pooled.
    )

    # Open one connection and close it automatically when this block ends.
    async with connectable.connect() as connection:
        # Run the regular synchronous migration function through SQLAlchemy's bridge.
        await connection.run_sync(do_run_migrations)

    # Dispose of the engine and release any remaining resources.
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations against the live database."""

    # Start and wait for the async migration workflow from normal synchronous code.
    asyncio.run(run_async_migrations())


# Alembic sets this flag based on whether the command requested SQL output only.
if context.is_offline_mode():
    run_migrations_offline()  # Generate SQL without connecting to PostgreSQL.
else:
    run_migrations_online()  # Connect to PostgreSQL and apply migrations.
