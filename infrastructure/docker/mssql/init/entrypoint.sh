#!/bin/bash
# Waits for SQL Server to accept connections, then applies the bootstrap script.
#
# The mssql/server image has no docker-entrypoint-initdb.d, so this runs as its
# own short-lived compose service. It reuses the server image because sqlcmd
# already ships there (/opt/mssql-tools18/bin).
set -euo pipefail

SQLCMD=/opt/mssql-tools18/bin/sqlcmd
HOST="${MSSQL_HOST:-mssql}"

for attempt in $(seq 1 60); do
    # -C trusts the server's self-signed certificate (local development only).
    if "$SQLCMD" -S "$HOST" -U sa -P "$MSSQL_SA_PASSWORD" -C -l 3 \
        -Q "SELECT 1" > /dev/null 2>&1; then
        echo "SQL Server reachable after ${attempt} attempt(s)."
        break
    fi
    if [ "$attempt" -eq 60 ]; then
        echo "SQL Server did not accept connections in time." >&2
        exit 1
    fi
    sleep 2
done

"$SQLCMD" -S "$HOST" -U sa -P "$MSSQL_SA_PASSWORD" -C -b \
    -v DB_NAME="${MSSQL_DB:-copilot}" \
    -v APP_LOGIN="${MSSQL_APP_USER:-copilot_app}" \
    -v APP_PASSWORD="${MSSQL_APP_PASSWORD}" \
    -v RO_LOGIN="${MSSQL_READONLY_USER:-copilot_readonly}" \
    -v RO_PASSWORD="${MSSQL_READONLY_PASSWORD}" \
    -i /init/01-bootstrap.sql
