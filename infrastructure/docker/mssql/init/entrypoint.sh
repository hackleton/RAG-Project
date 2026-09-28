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

# Reachable is not the same as ready. On a restart with an existing volume,
# SQL Server answers `SELECT 1` against master while the application database
# is still recovering, and the bootstrap then fails with Msg 904 ("cannot be
# autostarted during server shutdown or startup"). A fresh volume - which is
# what CI always has - never hits this, so it only ever bit a developer's
# second `docker compose up`. ABSENT means the database does not exist yet,
# which the bootstrap is about to fix; anything else must be ONLINE.
DB="${MSSQL_DB:-copilot}"
for attempt in $(seq 1 60); do
    state=$("$SQLCMD" -S "$HOST" -U sa -P "$MSSQL_SA_PASSWORD" -C -l 3 -h -1 -W \
        -Q "SET NOCOUNT ON; SELECT COALESCE(CONVERT(varchar(32), DATABASEPROPERTYEX('${DB}', 'Status')), 'ABSENT')" \
        2> /dev/null | tr -d '[:space:]' || true)
    if [ "$state" = "ONLINE" ] || [ "$state" = "ABSENT" ]; then
        echo "Database '${DB}' is ${state} after ${attempt} attempt(s)."
        break
    fi
    if [ "$attempt" -eq 60 ]; then
        echo "Database '${DB}' never came online (last state: ${state:-unknown})." >&2
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
