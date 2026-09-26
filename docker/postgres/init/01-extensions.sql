-- Runs once, on first initialization of an empty data volume.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy company-name matching
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()