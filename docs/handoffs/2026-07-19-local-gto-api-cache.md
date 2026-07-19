# Local-only GTO API cache migration (2026-07-19)

## Decision

`gto_api_cache` is removed from Supabase. Solver responses now use two local
layers only:

1. process memory for hot reads;
2. atomic JSON files under bind-mounted `.gto_cache` for durable reads.

This is the personal solve library described by North Star §7.9 and §9: it is
owner-controlled, exportable, survives deploys, and does not spend Supabase
database storage or shared-pooler bandwidth.

## Safe rollout

`scripts/deploy.sh` owns the destructive migration sequence:

1. resumably export/compare every DB row while the bot remains available;
2. stop the bot so no cache writer remains;
3. run the export/verification again to capture the final delta;
4. apply the migration that drops `public.gto_api_cache`;
5. build and start the local-only bot container.

The exit trap restarts the existing container if any post-stop step fails. The
export is idempotent, repairs corrupt files, and only reports success after each
local payload exactly matches its database row.

## Operational contract

- `.gto_cache` must remain bind-mounted and included in host backups.
- A missing/corrupt entry is a visible local miss and is re-fetched from GTOW.
- `/metrics` counts local cache files instead of querying Supabase.
- Do not reintroduce DB writes for solver cache payloads; portability is part of
  the data-sovereignty invariant.

## Migration evidence

Before the PR was opened, the production table snapshot was exported into the
host bind mount: **15,018 / 15,018 rows verified**. The Supabase relation used
536,485,888 bytes (508,506,398 bytes of JSON payload); the local JSON library is
larger because it is not PostgreSQL-compressed, so host disk/backup capacity is
the intentional trade-off for recovering free-tier database space.
