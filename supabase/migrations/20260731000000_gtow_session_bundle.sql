-- Browser-observed GTOW session bundle. Extension sync is the primary source
-- for access tokens; backend refresh may use the same user-scoped advisory
-- lock when access has actually expired.

ALTER TABLE public.users
  ADD COLUMN IF NOT EXISTS gto_access_token text,
  ADD COLUMN IF NOT EXISTS gto_access_token_fingerprint text,
  ADD COLUMN IF NOT EXISTS gto_access_token_iat timestamptz,
  ADD COLUMN IF NOT EXISTS gto_access_token_exp timestamptz,
  ADD COLUMN IF NOT EXISTS gto_client_id text,
  ADD COLUMN IF NOT EXISTS gto_session_observed_at timestamptz,
  ADD COLUMN IF NOT EXISTS gto_access_token_source text,
  ADD COLUMN IF NOT EXISTS gto_backend_signing_keypair jsonb;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'users_gto_access_token_source_check'
      AND conrelid = 'public.users'::regclass
  ) THEN
    ALTER TABLE public.users
      ADD CONSTRAINT users_gto_access_token_source_check
      CHECK (
        gto_access_token_source IS NULL
        OR gto_access_token_source IN ('extension', 'backend_refresh')
      );
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.sync_gtow_session_bundle(
  p_credential_hash text,
  p_refresh_token text,
  p_refresh_token_fingerprint text,
  p_refresh_token_iat timestamptz,
  p_refresh_token_exp timestamptz,
  p_access_token text,
  p_access_token_fingerprint text,
  p_access_token_iat timestamptz,
  p_access_token_exp timestamptz,
  p_client_id text,
  p_observed_at timestamptz,
  p_force boolean DEFAULT false
)
RETURNS TABLE(sync_result text, device_id uuid, synced_user_id bigint)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_user_id bigint;
  v_device public.gtow_sync_devices%ROWTYPE;
  v_existing_refresh_fingerprint text;
  v_existing_refresh_iat timestamptz;
  v_existing_access_fingerprint text;
  v_existing_access_iat timestamptz;
  v_existing_observed_at timestamptz;
  v_result text;
BEGIN
  IF p_credential_hash IS NULL
     OR length(p_credential_hash) <> 64
     OR p_refresh_token IS NULL
     OR p_refresh_token_fingerprint IS NULL
     OR length(p_refresh_token_fingerprint) <> 64
     OR p_access_token IS NULL
     OR p_access_token_fingerprint IS NULL
     OR length(p_access_token_fingerprint) <> 64
     OR p_refresh_token_iat IS NULL
     OR p_refresh_token_exp IS NULL
     OR p_access_token_iat IS NULL
     OR p_access_token_exp IS NULL
     OR p_observed_at IS NULL
     OR p_access_token_exp <= now() THEN
    RAISE EXCEPTION 'SESSION_BUNDLE_INVALID' USING ERRCODE = 'P0001';
  END IF;

  IF p_client_id IS NOT NULL
     AND (char_length(p_client_id) > 128
          OR p_client_id !~ '^[A-Za-z0-9._:-]*$') THEN
    RAISE EXCEPTION 'GTOW_CLIENT_ID_INVALID' USING ERRCODE = 'P0001';
  END IF;

  SELECT user_id INTO v_user_id
  FROM public.gtow_sync_devices
  WHERE credential_hash = p_credential_hash
    AND revoked_at IS NULL;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'DEVICE_UNAUTHORIZED' USING ERRCODE = 'P0001';
  END IF;

  PERFORM pg_advisory_xact_lock(hashtextextended('gtow-session:' || v_user_id::text, 0));

  SELECT
    gto_refresh_token_fingerprint,
    gto_refresh_token_iat,
    gto_access_token_fingerprint,
    gto_access_token_iat,
    gto_session_observed_at
  INTO
    v_existing_refresh_fingerprint,
    v_existing_refresh_iat,
    v_existing_access_fingerprint,
    v_existing_access_iat,
    v_existing_observed_at
  FROM public.users
  WHERE user_id = v_user_id
  FOR UPDATE;

  SELECT * INTO v_device
  FROM public.gtow_sync_devices
  WHERE credential_hash = p_credential_hash
    AND user_id = v_user_id
    AND revoked_at IS NULL
  FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'DEVICE_UNAUTHORIZED' USING ERRCODE = 'P0001';
  END IF;

  IF NOT p_force THEN
    IF v_existing_refresh_iat IS NOT NULL
       AND p_refresh_token_iat < v_existing_refresh_iat THEN
      RAISE EXCEPTION 'STALE_REFRESH_TOKEN' USING ERRCODE = 'P0001';
    END IF;

    IF v_existing_refresh_iat IS NOT NULL
       AND p_refresh_token_iat = v_existing_refresh_iat
       AND v_existing_refresh_fingerprint IS DISTINCT FROM p_refresh_token_fingerprint THEN
      RAISE EXCEPTION 'CONFLICTING_REFRESH_TOKEN' USING ERRCODE = 'P0001';
    END IF;
  END IF;

  IF v_existing_access_fingerprint IS DISTINCT FROM p_access_token_fingerprint THEN
    IF v_existing_access_iat IS NOT NULL
       AND p_access_token_iat < v_existing_access_iat THEN
      RAISE EXCEPTION 'STALE_ACCESS_TOKEN' USING ERRCODE = 'P0001';
    END IF;

    IF v_existing_access_iat IS NULL
       AND v_existing_observed_at IS NOT NULL
       AND p_observed_at < v_existing_observed_at THEN
      RAISE EXCEPTION 'STALE_ACCESS_TOKEN' USING ERRCODE = 'P0001';
    END IF;

    IF v_existing_access_iat IS NOT NULL
       AND p_access_token_iat = v_existing_access_iat
       AND v_existing_access_fingerprint IS DISTINCT FROM p_access_token_fingerprint THEN
      RAISE EXCEPTION 'CONFLICTING_ACCESS_TOKEN' USING ERRCODE = 'P0001';
    END IF;
  END IF;

  v_result := CASE
    WHEN v_existing_refresh_fingerprint = p_refresh_token_fingerprint
     AND v_existing_access_fingerprint = p_access_token_fingerprint THEN 'unchanged'
    ELSE 'updated'
  END;

  UPDATE public.users
  SET gto_refresh_token = p_refresh_token,
      gto_refresh_token_fingerprint = p_refresh_token_fingerprint,
      gto_refresh_token_iat = p_refresh_token_iat,
      gto_access_token = p_access_token,
      gto_access_token_fingerprint = p_access_token_fingerprint,
      gto_access_token_iat = p_access_token_iat,
      gto_access_token_exp = p_access_token_exp,
      gto_client_id = NULLIF(p_client_id, ''),
      gto_session_observed_at = p_observed_at,
      gto_access_token_source = 'extension',
      gto_token_synced_at = now()
  WHERE user_id = v_user_id;

  UPDATE public.gtow_sync_devices
  SET last_token_fingerprint = p_refresh_token_fingerprint,
      last_seen_at = now(),
      last_sync_at = now()
  WHERE id = v_device.id;

  INSERT INTO public.gtow_token_sync_events
    (user_id, device_id, token_fingerprint, token_exp, result)
  VALUES
    (v_user_id, v_device.id, p_access_token_fingerprint, p_access_token_exp, v_result);

  RETURN QUERY SELECT v_result, v_device.id, v_user_id;
END;
$$;

REVOKE ALL ON FUNCTION public.sync_gtow_session_bundle(
  text, text, text, timestamptz, timestamptz, text, text, timestamptz,
  timestamptz, text, timestamptz, boolean
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.sync_gtow_session_bundle(
  text, text, text, timestamptz, timestamptz, text, text, timestamptz,
  timestamptz, text, timestamptz, boolean
) TO service_role, postgres;
