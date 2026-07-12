-- Chrome Extension pairing + automatic GTO Wizard refresh-token sync.
-- Token bodies remain in users.gto_refresh_token for compatibility with the
-- existing Python bot. Audit tables store fingerprints only, never tokens.

ALTER TABLE public.users
  ADD COLUMN IF NOT EXISTS gto_refresh_token_fingerprint text,
  ADD COLUMN IF NOT EXISTS gto_token_synced_at timestamptz;

CREATE TABLE public.gtow_device_pairings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id bigint NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  code_hash text NOT NULL UNIQUE CHECK (length(code_hash) = 64),
  expires_at timestamptz NOT NULL,
  consumed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX gtow_device_pairings_user_created_idx
  ON public.gtow_device_pairings (user_id, created_at DESC);

CREATE TABLE public.gtow_sync_devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id bigint NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 80),
  credential_hash text NOT NULL UNIQUE CHECK (length(credential_hash) = 64),
  last_token_fingerprint text,
  last_seen_at timestamptz,
  last_sync_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX gtow_sync_devices_user_created_idx
  ON public.gtow_sync_devices (user_id, created_at DESC);

CREATE TABLE public.gtow_token_sync_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id bigint NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  device_id uuid REFERENCES public.gtow_sync_devices(id) ON DELETE SET NULL,
  token_fingerprint text NOT NULL,
  token_exp timestamptz,
  result text NOT NULL CHECK (result IN ('updated', 'unchanged')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX gtow_token_sync_events_user_created_idx
  ON public.gtow_token_sync_events (user_id, created_at DESC);

ALTER TABLE public.gtow_device_pairings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.gtow_sync_devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.gtow_token_sync_events ENABLE ROW LEVEL SECURITY;

-- Atomically consume a one-time pairing code and mint its device row. The raw
-- credential is returned by the Edge Function only; Postgres stores its HMAC.
CREATE OR REPLACE FUNCTION public.claim_gtow_device_pairing(
  p_code_hash text,
  p_credential_hash text,
  p_device_name text
)
RETURNS TABLE(device_id uuid, paired_user_id bigint, telegram_label text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_pairing public.gtow_device_pairings%ROWTYPE;
  v_device_id uuid;
BEGIN
  SELECT * INTO v_pairing
  FROM public.gtow_device_pairings
  WHERE code_hash = p_code_hash
  FOR UPDATE;

  IF NOT FOUND OR v_pairing.consumed_at IS NOT NULL OR v_pairing.expires_at <= now() THEN
    RAISE EXCEPTION 'PAIRING_INVALID_OR_EXPIRED' USING ERRCODE = 'P0001';
  END IF;

  IF char_length(trim(p_device_name)) NOT BETWEEN 1 AND 80 THEN
    RAISE EXCEPTION 'DEVICE_NAME_INVALID' USING ERRCODE = 'P0001';
  END IF;

  UPDATE public.gtow_device_pairings
  SET consumed_at = now()
  WHERE id = v_pairing.id;

  INSERT INTO public.gtow_sync_devices (user_id, name, credential_hash)
  VALUES (v_pairing.user_id, trim(p_device_name), p_credential_hash)
  RETURNING id INTO v_device_id;

  RETURN QUERY
  SELECT
    v_device_id,
    v_pairing.user_id,
    COALESCE(NULLIF('@' || u.username, '@'), NULLIF(u.name, ''), 'Telegram user')
  FROM public.users u
  WHERE u.user_id = v_pairing.user_id;
END;
$$;

-- Authenticate a device by credential HMAC, update the user's current token,
-- and record only non-secret metadata in the audit trail.
CREATE OR REPLACE FUNCTION public.sync_gtow_refresh_token(
  p_credential_hash text,
  p_refresh_token text,
  p_token_fingerprint text,
  p_token_exp timestamptz
)
RETURNS TABLE(sync_result text, device_id uuid, synced_user_id bigint)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_device public.gtow_sync_devices%ROWTYPE;
  v_existing_fingerprint text;
  v_result text;
BEGIN
  SELECT * INTO v_device
  FROM public.gtow_sync_devices
  WHERE credential_hash = p_credential_hash
    AND revoked_at IS NULL
  FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'DEVICE_UNAUTHORIZED' USING ERRCODE = 'P0001';
  END IF;

  SELECT gto_refresh_token_fingerprint INTO v_existing_fingerprint
  FROM public.users
  WHERE user_id = v_device.user_id
  FOR UPDATE;

  v_result := CASE
    WHEN v_existing_fingerprint = p_token_fingerprint THEN 'unchanged'
    ELSE 'updated'
  END;

  UPDATE public.users
  SET gto_refresh_token = p_refresh_token,
      gto_refresh_token_fingerprint = p_token_fingerprint,
      gto_token_synced_at = now()
  WHERE user_id = v_device.user_id;

  UPDATE public.gtow_sync_devices
  SET last_token_fingerprint = p_token_fingerprint,
      last_seen_at = now(),
      last_sync_at = now()
  WHERE id = v_device.id;

  INSERT INTO public.gtow_token_sync_events
    (user_id, device_id, token_fingerprint, token_exp, result)
  VALUES
    (v_device.user_id, v_device.id, p_token_fingerprint, p_token_exp, v_result);

  RETURN QUERY SELECT v_result, v_device.id, v_device.user_id;
END;
$$;

REVOKE ALL ON public.gtow_device_pairings FROM anon, authenticated;
REVOKE ALL ON public.gtow_sync_devices FROM anon, authenticated;
REVOKE ALL ON public.gtow_token_sync_events FROM anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_gtow_device_pairing(text, text, text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.sync_gtow_refresh_token(text, text, text, timestamptz) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_gtow_device_pairing(text, text, text) TO service_role, postgres;
GRANT EXECUTE ON FUNCTION public.sync_gtow_refresh_token(text, text, text, timestamptz) TO service_role, postgres;
