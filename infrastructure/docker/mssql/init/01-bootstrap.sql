-- Database, application login, and the restricted login the SQL agent uses.
--
-- Idempotent: re-running is a no-op, so the init service can be left in the
-- compose file and safely re-run on every `docker compose up`.
--
-- Passwords arrive as sqlcmd variables (-v), never as literals in this file.
SET NOCOUNT ON;
GO

IF DB_ID('$(DB_NAME)') IS NULL
BEGIN
    PRINT 'Creating database $(DB_NAME)';
    EXEC('CREATE DATABASE [$(DB_NAME)]');
END
GO

-- READ_COMMITTED_SNAPSHOT: readers do not block writers. Without it the SQL
-- agent's analytical SELECTs take shared locks that stall document ingestion.
IF EXISTS (SELECT 1 FROM sys.databases
           WHERE name = '$(DB_NAME)' AND is_read_committed_snapshot_on = 0)
BEGIN
    PRINT 'Enabling READ_COMMITTED_SNAPSHOT';
    EXEC('ALTER DATABASE [$(DB_NAME)] SET READ_COMMITTED_SNAPSHOT ON WITH ROLLBACK IMMEDIATE');
END
GO

-- ---------------------------------------------------------------------------
-- Application login: owns the schema, runs migrations.
-- ---------------------------------------------------------------------------
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = '$(APP_LOGIN)')
BEGIN
    PRINT 'Creating login $(APP_LOGIN)';
    EXEC('CREATE LOGIN [$(APP_LOGIN)] WITH PASSWORD = ''$(APP_PASSWORD)'',
          DEFAULT_DATABASE = [$(DB_NAME)], CHECK_POLICY = OFF');
END
GO

USE [$(DB_NAME)];
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = '$(APP_LOGIN)')
    EXEC('CREATE USER [$(APP_LOGIN)] FOR LOGIN [$(APP_LOGIN)]');
GO
EXEC('ALTER ROLE db_owner ADD MEMBER [$(APP_LOGIN)]');
GO

-- ---------------------------------------------------------------------------
-- Agent login: SELECT only (project plan section 10).
--
-- The agent never connects as the application. db_datareader covers every
-- table that exists now and every table a later migration adds, so a new table
-- cannot silently become readable-and-writable. The DENYs are belt and braces:
-- a DENY outranks any GRANT, including one picked up from a future role.
-- ---------------------------------------------------------------------------
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = '$(RO_LOGIN)')
BEGIN
    PRINT 'Creating login $(RO_LOGIN)';
    EXEC('CREATE LOGIN [$(RO_LOGIN)] WITH PASSWORD = ''$(RO_PASSWORD)'',
          DEFAULT_DATABASE = [$(DB_NAME)], CHECK_POLICY = OFF');
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = '$(RO_LOGIN)')
    EXEC('CREATE USER [$(RO_LOGIN)] FOR LOGIN [$(RO_LOGIN)]');
GO

EXEC('ALTER ROLE db_datareader ADD MEMBER [$(RO_LOGIN)]');
GO

DENY INSERT, UPDATE, DELETE, ALTER, REFERENCES, EXECUTE TO [$(RO_LOGIN)];
GO
DENY CREATE TABLE, CREATE VIEW, CREATE PROCEDURE, CREATE FUNCTION TO [$(RO_LOGIN)];
GO
-- Deny the catalog surface an injected statement would use to enumerate the
-- schema beyond what db_datareader already implies.
DENY VIEW DEFINITION TO [$(RO_LOGIN)];
GO

PRINT 'Bootstrap complete: $(DB_NAME), $(APP_LOGIN) (db_owner), $(RO_LOGIN) (SELECT only).';
GO
