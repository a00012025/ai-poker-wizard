-- Harden token freshness and make sync/logout use the same lock order:
-- users row first, then the device row. This prevents logout races and
-- rejects an older refresh JWT after a newer one has been accepted.

ALTER TABLE public.users
  ADD COLUMN IF NOT EXISTS gto_refresh_token_iat timestamptz;

CREATE OR REPLACE FUNCTION public.sync_gtow_refresh_token(
  p_credential_hash text,
  p_refresh_token text,
  p_token_fingerprint text,
  p_token_iat timestamptz,
  p_token_exp timestamptz
)
RETURNS TABLE(sync_result text, device_id uuid, synced_user_id bigint)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_user_id bigint;
  v_device public.gtow_sync_devices%ROWTYPE;
  v_existing_fingerprint text;
  v_existing_iat timestamptz;
  v_result text;
BEGIN
  SELECT user_id INTO v_user_id
  FROM public.gtow_sync_devices
  WHERE credential_hash = p_credential_hash
    AND revoked_at IS NULL;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'DEVICE_UNAUTHORIZED' USING ERRCODE = 'P0001';
  END IF;

  SELECT gto_refresh_token_fingerprint, gto_refresh_token_iat
  INTO v_existing_fingerprint, v_existing_iat
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

  IF v_existing_iat IS NOT NULL
     AND p_token_iat < v_existing_iat THEN
    RAISE EXCEPTION 'STALE_REFRESH_TOKEN' USING ERRCODE = 'P0001';
  END IF;

  IF v_existing_iat IS NOT NULL
     AND p_token_iat = v_existing_iat
     AND v_existing_fingerprint IS DISTINCT FROM p_token_fingerprint THEN
    RAISE EXCEPTION 'CONFLICTING_REFRESH_TOKEN' USING ERRCODE = 'P0001';
  END IF;

  v_result := CASE
    WHEN v_existing_fingerprint = p_token_fingerprint THEN 'unchanged'
    ELSE 'updated'
  END;

  UPDATE public.users
  SET gto_refresh_token = p_refresh_token,
      gto_refresh_token_fingerprint = p_token_fingerprint,
      gto_refresh_token_iat = p_token_iat,
      gto_token_synced_at = now()
  WHERE user_id = v_user_id;

  UPDATE public.gtow_sync_devices
  SET last_token_fingerprint = p_token_fingerprint,
      last_seen_at = now(),
      last_sync_at = now()
  WHERE id = v_device.id;

  INSERT INTO public.gtow_token_sync_events
    (user_id, device_id, token_fingerprint, token_exp, result)
  VALUES
    (v_user_id, v_device.id, p_token_fingerprint, p_token_exp, v_result);

  RETURN QUERY SELECT v_result, v_device.id, v_user_id;
END;
$$;

DROP FUNCTION IF EXISTS public.sync_gtow_refresh_token(text, text, text, timestamptz);
REVOKE ALL ON FUNCTION public.sync_gtow_refresh_token(text, text, text, timestamptz, timestamptz)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.sync_gtow_refresh_token(text, text, text, timestamptz, timestamptz)
  TO service_role, postgres;

-- Keep operational metadata bounded without requiring a separate scheduler.
CREATE OR REPLACE FUNCTION public.cleanup_gtow_token_sync_metadata()
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  DELETE FROM public.gtow_device_pairings
  WHERE created_at < now() - interval '7 days';
  DELETE FROM public.gtow_token_sync_events
  WHERE created_at < now() - interval '180 days';
$$;

REVOKE ALL ON FUNCTION public.cleanup_gtow_token_sync_metadata()
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.cleanup_gtow_token_sync_metadata()
  TO service_role, postgres;
