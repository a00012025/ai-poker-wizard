#!/usr/bin/env python3
"""Export the legacy Supabase GTO cache into the persistent local solve library.

The export is resumable and idempotent: matching files are retained, missing or
corrupt files are atomically replaced, and every database row is compared with
its final local representation before success is reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psycopg2


TABLE = "public.gto_api_cache"
MANIFEST = "export_manifest.json"


def _payload(response, is_null: bool) -> dict:
    return {"is_null": True} if is_null else {
        "is_null": False,
        "response": response,
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(data, f, allow_nan=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _sync_row(output_dir: Path, cache_key: str, response, is_null: bool) -> bool:
    """Make one local row exact; return True when a file had to be written."""
    expected = _payload(response, is_null)
    path = output_dir / f"{cache_key}.json"
    if _read_json(path) == expected:
        return False
    _atomic_json(path, expected)
    if _read_json(path) != expected:
        raise RuntimeError(f"post-write verification failed: {path}")
    return True


def _connect():
    dsn = os.getenv("SUPABASE_CONN")
    if not dsn:
        raise RuntimeError("SUPABASE_CONN environment variable not set")
    return psycopg2.connect(dsn, connect_timeout=15)


def table_exists() -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (TABLE,))
        return bool(cur.fetchone()[0])


def export(output_dir: Path, batch_size: int = 100) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL", (TABLE,))
            if not cur.fetchone()[0]:
                return {"table_exists": False, "rows": 0, "written": 0}

        rows = written = 0
        with conn.cursor(name="gto_api_cache_export") as cur:
            cur.itersize = batch_size
            cur.execute(
                "SELECT cache_key, response, is_null "
                f"FROM {TABLE} ORDER BY cache_key"
            )
            for cache_key, response, is_null in cur:
                written += int(_sync_row(output_dir, cache_key, response, is_null))
                rows += 1
                if rows % 500 == 0:
                    print(f"exported/verified {rows} rows", flush=True)

    manifest = {
        "source": TABLE,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "verified": True,
    }
    _atomic_json(output_dir / MANIFEST, manifest)
    return {"table_exists": True, "rows": rows, "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".gto_cache"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--table-exists", action="store_true",
        help="only check whether the legacy table still exists (0=yes, 1=no)",
    )
    args = parser.parse_args()

    try:
        if args.table_exists:
            return 0 if table_exists() else 1
        result = export(args.output_dir, max(1, args.batch_size))
        if not result["table_exists"]:
            print("gto_api_cache is already absent; nothing to export")
            return 0
        print(
            f"verified {result['rows']} rows in {args.output_dir} "
            f"({result['written']} files written)"
        )
        return 0
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
