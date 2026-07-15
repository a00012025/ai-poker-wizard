-- Extension-triggered hand ingest queue. Rows are created by the gtow-sync
-- Edge Function (device-authenticated) and consumed by the bot's 5s poller.
-- No tokens are stored here — the runner reads users.gto_refresh_token.

CREATE TABLE public.gtow_ingest_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id bigint NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  device_id uuid REFERENCES public.gtow_sync_devices(id) ON DELETE SET NULL,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'done', 'error')),
  progress text,
  result text,
  requested_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  heartbeat_at timestamptz,        -- refreshed on every progress write
  finished_at timestamptz
);

CREATE INDEX gtow_ingest_requests_pending_idx
  ON public.gtow_ingest_requests (requested_at)
  WHERE status IN ('pending', 'running');
CREATE INDEX gtow_ingest_requests_user_idx
  ON public.gtow_ingest_requests (user_id, requested_at DESC);
-- At most one open request per user; makes enqueue dedupe atomic
-- (targetless ON CONFLICT DO NOTHING catches this partial index too).
CREATE UNIQUE INDEX gtow_ingest_requests_one_open_per_user_idx
  ON public.gtow_ingest_requests (user_id)
  WHERE status IN ('pending', 'running');

ALTER TABLE public.gtow_ingest_requests ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.gtow_ingest_requests FROM anon, authenticated;

-- Device-authenticated enqueue: reuse the caller's open request instead of
-- stacking duplicates. Runs as service_role via the Edge Function only.
CREATE OR REPLACE FUNCTION public.enqueue_gtow_ingest_request(
  p_credential_hash text
)
RETURNS TABLE(request_id uuid, request_status text, reused boolean)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_device public.gtow_sync_devices%ROWTYPE;
  v_existing public.gtow_ingest_requests%ROWTYPE;
  v_id uuid;
BEGIN
  SELECT * INTO v_device
  FROM public.gtow_sync_devices
  WHERE credential_hash = p_credential_hash
    AND revoked_at IS NULL;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'DEVICE_UNAUTHORIZED' USING ERRCODE = 'P0001';
  END IF;

  -- Atomic dedupe: the partial unique index (one open request per user)
  -- turns a concurrent double-trigger into a no-op insert we then re-read.
  INSERT INTO public.gtow_ingest_requests (user_id, device_id)
  VALUES (v_device.user_id, v_device.id)
  ON CONFLICT DO NOTHING
  RETURNING id INTO v_id;

  IF v_id IS NOT NULL THEN
    RETURN QUERY SELECT v_id, 'pending'::text, false;
    RETURN;
  END IF;

  SELECT * INTO v_existing
  FROM public.gtow_ingest_requests
  WHERE user_id = v_device.user_id
    AND status IN ('pending', 'running')
  ORDER BY requested_at
  LIMIT 1;

  IF NOT FOUND THEN
    -- The conflicting row closed between our INSERT and re-read; retry once.
    INSERT INTO public.gtow_ingest_requests (user_id, device_id)
    VALUES (v_device.user_id, v_device.id)
    ON CONFLICT DO NOTHING
    RETURNING id INTO v_id;
    IF v_id IS NULL THEN
      RAISE EXCEPTION 'INGEST_ENQUEUE_RACE' USING ERRCODE = 'P0001';
    END IF;
    RETURN QUERY SELECT v_id, 'pending'::text, false;
    RETURN;
  END IF;

  RETURN QUERY SELECT v_existing.id, v_existing.status, true;
END;
$$;

-- Device-authenticated status read, scoped to the device's own user.
CREATE OR REPLACE FUNCTION public.get_gtow_ingest_request(
  p_credential_hash text,
  p_request_id uuid
)
RETURNS TABLE(request_status text, progress text, result text,
              requested_at timestamptz, finished_at timestamptz)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_device public.gtow_sync_devices%ROWTYPE;
BEGIN
  SELECT * INTO v_device
  FROM public.gtow_sync_devices
  WHERE credential_hash = p_credential_hash
    AND revoked_at IS NULL;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'DEVICE_UNAUTHORIZED' USING ERRCODE = 'P0001';
  END IF;

  RETURN QUERY
  SELECT r.status, r.progress, r.result, r.requested_at, r.finished_at
  FROM public.gtow_ingest_requests r
  WHERE r.id = p_request_id
    AND r.user_id = v_device.user_id;
END;
$$;

REVOKE ALL ON FUNCTION public.enqueue_gtow_ingest_request(text)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_gtow_ingest_request(text, uuid)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.enqueue_gtow_ingest_request(text)
  TO service_role, postgres;
GRANT EXECUTE ON FUNCTION public.get_gtow_ingest_request(text, uuid)
  TO service_role, postgres;

-- Manual sync/ingest triggers force-override the stored token: the user just
-- proved the session works by being logged in, while the "newer" stored iat
-- may belong to a FORCED_LOGOUT-dead token. Freshness guards only make sense
-- for the passive auto-sync path (p_force = false).
CREATE OR REPLACE FUNCTION public.sync_gtow_refresh_token(
  p_credential_hash text,
  p_refresh_token text,
  p_token_fingerprint text,
  p_token_iat timestamptz,
  p_token_exp timestamptz,
  p_force boolean
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

  IF NOT p_force THEN
    IF v_existing_iat IS NOT NULL
       AND p_token_iat < v_existing_iat THEN
      RAISE EXCEPTION 'STALE_REFRESH_TOKEN' USING ERRCODE = 'P0001';
    END IF;

    IF v_existing_iat IS NOT NULL
       AND p_token_iat = v_existing_iat
       AND v_existing_fingerprint IS DISTINCT FROM p_token_fingerprint THEN
      RAISE EXCEPTION 'CONFLICTING_REFRESH_TOKEN' USING ERRCODE = 'P0001';
    END IF;
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

-- Keep the 5-arg signature as a thin wrapper so the already-deployed Edge
-- Function keeps working until the new one ships (no-downtime rollout).
CREATE OR REPLACE FUNCTION public.sync_gtow_refresh_token(
  p_credential_hash text,
  p_refresh_token text,
  p_token_fingerprint text,
  p_token_iat timestamptz,
  p_token_exp timestamptz
)
RETURNS TABLE(sync_result text, device_id uuid, synced_user_id bigint)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT * FROM public.sync_gtow_refresh_token(
    p_credential_hash, p_refresh_token, p_token_fingerprint,
    p_token_iat, p_token_exp, false);
$$;

REVOKE ALL ON FUNCTION public.sync_gtow_refresh_token(text, text, text, timestamptz, timestamptz, boolean)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.sync_gtow_refresh_token(text, text, text, timestamptz, timestamptz, boolean)
  TO service_role, postgres;
