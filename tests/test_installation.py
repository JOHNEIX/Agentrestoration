import importlib.util

import pytest

from agentrestoration.core import ROOT


def assets_module():
    spec = importlib.util.spec_from_file_location("setup_assets", ROOT / "scripts/setup_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["retinexformer", "dehazeformer"])
def test_asset_reuse_and_corrupt_weight_repair(tmp_path, monkeypatch, name):
    module = assets_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = module.SOURCES[name]
    fetched = []
    def fetch(url, destination):
        fetched.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = b"weights" if url == source["url"] else b"return coordw.detach(), coordh.detach()"
        destination.write_bytes(content)
    monkeypatch.setattr(module, "fetch", fetch)
    assert not module.assets_ready(name)
    module.image_tool(name)
    assert module.assets_ready(name)
    monkeypatch.setattr("sys.argv", ["setup_assets.py", "--tool", name])
    before = len(fetched)
    module.main()
    assert len(fetched) == before  # repeated installation works without network
    weights = tmp_path / "weights" / source["weights"]
    weights.write_bytes(b"interrupted")
    assert not module.assets_ready(name)
    module.main()
    assert weights.read_bytes() == b"weights"
    assert fetched.count(source["url"]) == 2


def test_asset_check_never_downloads(tmp_path, monkeypatch):
    module = assets_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "fetch", lambda *args: pytest.fail("check must stay offline"))
    monkeypatch.setattr("sys.argv", ["setup_assets.py", "--check"])
    with pytest.raises(SystemExit, match="restormer: missing"):
        module.main()


def test_api_example_format_and_local_file(tmp_path, monkeypatch):
    from agentrestoration import adapters
    monkeypatch.setattr(adapters, "ROOT", tmp_path)
    for key in ("VLM_API_KEY", "IMAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    sample = (ROOT / "api.example.txt").read_text(encoding="utf-8")
    (tmp_path / "api.txt").write_text(sample, encoding="utf-8")
    assert adapters.credentials() == {"vlm": "", "image": ""}
    lines = sample.splitlines()
    lines.extend([""] * (4 - len(lines)))
    lines[1] = "test-vision-token"
    lines[3] = "test-edit-token"
    (tmp_path / "api.txt").write_text("\n".join(lines), encoding="utf-8")
    assert adapters.credentials() == {"vlm": "test-vision-token", "image": "test-edit-token"}
