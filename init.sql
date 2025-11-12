-- Initialize GSL Database
-- This script sets up the initial database structure and sample data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for text search (will be used by GSL dictionary)
-- Additional setup will be done by Alembic migrations