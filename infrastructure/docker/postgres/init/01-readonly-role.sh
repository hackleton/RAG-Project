#!/bin/bash
# Restricted role for the SQL agent (project plan section 10).
#
# The agent never connects as the application owner. This role may only run
# SELECT, and the ALTER DEFAULT PRIVILEGES clause covers tables created by
# later migrations - so a new table cannot silently become writable.
#
# Runs once, on first initialisation of the data volume. A shell script rather
# than plain .sql so the password can come from the environment instead of
# being committed.
set -euo pipefail

READONLY_PASSWORD="${POSTGRES_READONLY_PASSWORD:-copilot_readonly_local_password}"

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v readonly_password="$READONLY_PASSWORD" <<'SQL'
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'copilot_readonly') THEN
        CREATE ROLE copilot_readonly LOGIN;
    END IF;
END
$$;

ALTER ROLE copilot_readonly WITH PASSWORD :'readonly_password';
-- Belt and braces: even a SELECT-only grant cannot be escaped into a write.
ALTER ROLE copilot_readonly SET default_transaction_read_only = on;
ALTER ROLE copilot_readonly SET statement_timeout = '10s';
ALTER ROLE copilot_readonly SET idle_in_transaction_session_timeout = '30s';

GRANT USAGE ON SCHEMA public TO copilot_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO copilot_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO copilot_readonly;
REVOKE CREATE ON SCHEMA public FROM copilot_readonly;
SQL

# GRANT CONNECT needs the database name, which is only known at runtime.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -c "GRANT CONNECT ON DATABASE \"$POSTGRES_DB\" TO copilot_readonly;"

echo "copilot_readonly role provisioned (SELECT-only, read-only transactions)."
