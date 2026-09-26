#!/bin/bash
# Creates the read-only role used by the LLM agent's tools.
# Password comes from the AGENT_RO_PASSWORD env var (never hard-coded).
set -euo pipefail

: "${AGENT_RO_PASSWORD:?AGENT_RO_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 \
     -v ro_pw="$AGENT_RO_PASSWORD" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-'EOSQL'
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_ro') THEN
            CREATE ROLE agent_ro LOGIN;
        END IF;
    END
    $$;

    ALTER ROLE agent_ro WITH PASSWORD :'ro_pw';
    ALTER ROLE agent_ro SET default_transaction_read_only = on;
    ALTER ROLE agent_ro SET statement_timeout = '10s';

    -- Grant CONNECT on whichever database this session is connected to.
    SELECT format('GRANT CONNECT ON DATABASE %I TO agent_ro', current_database()) \gexec

    GRANT USAGE ON SCHEMA public TO agent_ro;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM agent_ro;
EOSQL