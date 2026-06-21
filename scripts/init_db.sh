#!/usr/bin/env bash
# Run once after postgres is up to create the langfuse database
# Usage: ./scripts/init_db.sh

set -euo pipefail

POSTGRES_USER=${POSTGRES_USER:-pyrag}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-pyrag}
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

echo "Creating langfuse database if not exists..."

PGPASSWORD="$POSTGRES_PASSWORD" psql \
  -h "$POSTGRES_HOST" \
  -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" \
  -tc "SELECT 1 FROM pg_database WHERE datname = 'langfuse'" \
  | grep -q 1 || \
  PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$POSTGRES_HOST" \
    -p "$POSTGRES_PORT" \
    -U "$POSTGRES_USER" \
    -c "CREATE DATABASE langfuse;"

echo "Done."
