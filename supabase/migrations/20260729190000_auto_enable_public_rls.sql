-- Fail closed for every table created in Supabase's exposed public schema.
--
-- This guardrail is deliberately enforced by PostgreSQL rather than by code
-- review convention. If a future migration omits ENABLE ROW LEVEL SECURITY,
-- the event trigger adds it before CREATE TABLE completes. If the hardening
-- cannot be applied, the table creation fails instead of leaving data exposed.

BEGIN;

CREATE SCHEMA IF NOT EXISTS private;
REVOKE ALL ON SCHEMA private FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION private.auto_secure_public_table()
RETURNS EVENT_TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
DECLARE
  cmd record;
BEGIN
  FOR cmd IN
    SELECT *
    FROM pg_event_trigger_ddl_commands()
    WHERE command_tag IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
      AND object_type IN ('table', 'partitioned table')
  LOOP
    IF cmd.schema_name = 'public' THEN
      EXECUTE format(
        'ALTER TABLE IF EXISTS %s ENABLE ROW LEVEL SECURITY',
        cmd.object_identity
      );
      EXECUTE format(
        'REVOKE ALL ON TABLE %s FROM anon, authenticated',
        cmd.object_identity
      );
    END IF;
  END LOOP;
END;
$$;

REVOKE ALL ON FUNCTION private.auto_secure_public_table()
  FROM PUBLIC, anon, authenticated;

DROP EVENT TRIGGER IF EXISTS ensure_public_table_rls;
CREATE EVENT TRIGGER ensure_public_table_rls
ON ddl_command_end
WHEN TAG IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
EXECUTE FUNCTION private.auto_secure_public_table();

COMMIT;
