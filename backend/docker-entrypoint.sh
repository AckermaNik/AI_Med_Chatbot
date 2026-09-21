#!/usr/bin/env sh
set -eu

# The database is marked healthy by Compose before this container starts.
# Migrations are idempotent, so this also works with the existing pgdata volume.
alembic upgrade head

exec "$@"
