"""Deployment and local CLI integration regressions."""

import os
import subprocess
import tempfile
from pathlib import Path

from .harness import SCRIPTS_DIR, assert_eq, assert_in, assert_true


def test_deploy_resolves_supabase_shim_companion_and_rejects_version_drift():
    """Deploys must not depend on a manually exported Supabase binary path.

import pytest

pytestmark = pytest.mark.integration

    Regression for the 2026-07-13 production deploy: the `supabase` shim was
    installed in ~/.local/bin while `supabase-go` lived in
    ~/.local/share/supabase, so `supabase db push` stopped before deployment.
    """
    helper = SCRIPTS_DIR / "supabase_cli.sh"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bin_dir = root / "bin"
        shared_dir = root / "home" / ".local" / "share" / "supabase"
        bin_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)

        shim = bin_dir / "supabase"
        go = shared_dir / "supabase-go"
        shim.write_text("#!/bin/sh\necho 2.109.1\n")
        go.write_text("#!/bin/sh\necho 2.109.1\n")
        shim.chmod(0o755)
        go.chmod(0o755)

        env = {
            **os.environ,
            "HOME": str(root / "home"),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
        }
        env.pop("SUPABASE_GO_BINARY", None)
        command = (
            f'source "{helper}"; configure_supabase_cli || exit $?; '
            'printf "%s" "$SUPABASE_GO_BINARY"'
        )
        ok = subprocess.run(
            ["bash", "-c", command], env=env, text=True,
            capture_output=True, check=False,
        )
        assert_eq(ok.returncode, 0, ok.stderr)
        assert_eq(ok.stdout, str(go), "shared fallback must be exported")

        go.write_text("#!/bin/sh\necho 2.108.0\n")
        mismatch = subprocess.run(
            ["bash", "-c", command], env=env, text=True,
            capture_output=True, check=False,
        )
        assert_true(mismatch.returncode != 0, "version drift must fail closed")
        assert_in("版本不一致", mismatch.stderr)

        go.unlink()
        shim.write_text(
            "#!/bin/sh\n"
            "# supabase/legacy/LegacyGoProxy\n"
            "echo 2.109.1\n"
        )
        missing = subprocess.run(
            ["bash", "-c", command], env=env, text=True,
            capture_output=True, check=False,
        )
        assert_true(missing.returncode != 0, "a companion-less shim must fail before deploy")
        assert_in("找不到 supabase-go", missing.stderr)
