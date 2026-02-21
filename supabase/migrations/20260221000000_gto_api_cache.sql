CREATE TABLE gto_api_cache (
    cache_key TEXT PRIMARY KEY,
    response JSONB,
    is_null BOOLEAN NOT NULL DEFAULT FALSE
);
