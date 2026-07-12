import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "chrome-extension"


def test_manifest_v2_sync_contract():
    manifest = json.loads((EXT / "manifest.json").read_text())
    assert manifest["version"] == "2.0.0"
    assert manifest["action"]["default_popup"] == "popup.html"
    assert "storage" in manifest["permissions"]
    assert "https://app.gtowizard.com/*" in manifest["host_permissions"]
    assert any("supabase.co" in host for host in manifest["host_permissions"])


def test_manifest_referenced_files_exist():
    manifest = json.loads((EXT / "manifest.json").read_text())
    referenced = {
        manifest["background"]["service_worker"],
        manifest["action"]["default_popup"],
    }
    for content_script in manifest["content_scripts"]:
        referenced.update(content_script["js"])
    for path in referenced:
        assert (EXT / path).is_file(), path


def test_extension_does_not_log_raw_tokens():
    for path in EXT.glob("*.js"):
        source = path.read_text()
        assert "console.log" not in source
        assert "chrome.storage.local.set({ refreshToken" not in source
