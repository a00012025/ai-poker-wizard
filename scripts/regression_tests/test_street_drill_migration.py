"""Exercise migration behavior in a disposable local PostgreSQL container."""
import asyncio
from pathlib import Path
import shutil
import subprocess
import uuid

import asyncpg
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_street_migration_upgrades_identity_and_accepts_new_urls():
    if not shutil.which('docker') or subprocess.run(
            ['docker', 'image', 'inspect', 'postgres:12'], capture_output=True).returncode:
        pytest.skip('local postgres:12 Docker image unavailable')
    name = 'street-migration-' + uuid.uuid4().hex[:12]
    subprocess.run(['docker','run','-d','--rm','--name',name,
                    '-e','POSTGRES_HOST_AUTH_METHOD=trust','-p','127.0.0.1::5432','postgres:12'],
                   check=True,capture_output=True)
    async def check():
        port = subprocess.check_output(['docker','port',name,'5432'],text=True).strip().rsplit(':',1)[1]
        for attempt in range(60):
            try:
                conn = await asyncpg.connect(host='127.0.0.1',port=int(port),user='postgres',database='postgres')
                break
            except (OSError, asyncpg.PostgresError):
                if attempt == 59:
                    raise
                await asyncio.sleep(.25)
        try:
            await conn.execute('''CREATE TABLE drill_queue (
                kind text, drill_url text, gtow_settings_hash text,
                gtow_drill_synced_at timestamptz, gtow_training_started_at timestamptz,
                gtow_baseline_totals jsonb)''')
            await conn.execute((ROOT/'supabase/migrations/20260716233000_full_hand_drills.sql').read_text())
            old = 'https://app.gtowizard.com/practice/trainer?fh_trainer_mode=stop_end_of_hand&fh_hero=BTN'
            new = old.replace('stop_end_of_hand','stop_end_of_street')
            await conn.execute("INSERT INTO drill_queue VALUES ('drill',$1,'old',now(),now(),'{}')",old)
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("INSERT INTO drill_queue(kind,drill_url) VALUES ('drill',$1)",new)
            await conn.execute((ROOT/'supabase/migrations/20261031160000_street_drills.sql').read_text())
            row = await conn.fetchrow('SELECT * FROM drill_queue')
            assert row['drill_url'] == new
            assert all(row[key] is None for key in ('gtow_settings_hash','gtow_drill_synced_at',
                                                   'gtow_training_started_at','gtow_baseline_totals'))
            await conn.execute("INSERT INTO drill_queue(kind,drill_url) VALUES ('drill',$1)",new)
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("INSERT INTO drill_queue(kind,drill_url) VALUES ('drill',$1)",old)
        finally:
            await conn.close()
    try:
        asyncio.run(check())
    finally:
        subprocess.run(['docker','rm','-f',name],check=True,capture_output=True)
