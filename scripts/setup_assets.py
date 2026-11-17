"""Download upstream assets into this workspace; no credentials involved."""
import json
import argparse
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentrestoration.core import ROOT, digest, write_json

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fetch(url, path)
    print(path.name, digest(path), flush=True)

def restormer():
    sha="68dc6ac472db26f16361150cb7a96a1bc87da93f"
    root=ROOT / "third_party/Restormer"
    download(f"https://raw.githubusercontent.com/swz30/Restormer/{sha}/basicsr/models/archs/restormer_arch.py",root / "basicsr/models/archs/restormer_arch.py")
    download(f"https://raw.githubusercontent.com/swz30/Restormer/{sha}/LICENSE.md",root / "LICENSE.md")
    write_json(root / "source.json",dict(repository="https://github.com/swz30/Restormer",commit=sha))
    download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",ROOT / "weights/yolo11n.pt")
    dest=ROOT / "weights/gaussian_color_denoising_blind.pth"
    download("https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth",dest)
    write_json(ROOT / "weights/sources.json",dict(restormer="https://github.com/swz30/Restormer/releases/tag/v1.0",
        sha256=digest(dest),architecture_commit=sha))

SOURCES = {
    "retinexformer": dict(repository="https://github.com/caiyuanhao1998/Retinexformer",
        commit="master", author="caiyuanhao1998", repo="Retinexformer",
        source="basicsr/models/archs/RetinexFormer_arch.py", local="RetinexFormer_arch.py", license="LICENSE.txt",
        directory="Retinexformer", weights="retinexformer_lol_v1.pth",
        url="https://huggingface.co/CaiYuanhao/Retinexformer/resolve/7983bc70dca2cf043000f1efe793d51172d2f83b/LOL_v1.pth"),
    "dehazeformer": dict(repository="https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo",
        commit="main", author="IDKiro", repo="DehazeFormer_Demo",
        source="models/dehazeformer.py", local="dehazeformer.py", license="LICENSE",
        directory="DehazeFormer", weights="dehazeformer_mct_mixed.pth",
        url="https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo/resolve/main/saved_models/dehazeformer.pth"),
}


def fetch(url, destination):
    import httpx
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=180) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
            temporary.replace(destination)
            return
        except httpx.HTTPError:
            if attempt == 2:
                raise
            time.sleep(1)


def image_tool(name):
    source = SOURCES[name]
    root = ROOT / "third_party" / source["directory"]
    raw = f"https://raw.githubusercontent.com/{source['author']}/{source['repo']}/{source['commit']}"
    if name == "dehazeformer":
        raw = f"{source['repository']}/resolve/{source['commit']}"
    architecture = root / source["local"]
    fetch(f"{raw}/{source['source']}", architecture)
    original_sha256 = digest(architecture)
    compatibility = None
    if name == "dehazeformer":
        text = architecture.read_text(encoding="utf-8")
        old = "return coordw.detach(), coordh.detach()"
        if old not in text:
            raise ValueError("DehazeFormer source import changed")
        architecture.write_text(text.replace(old,
            "return coordw.detach().to(x), coordh.detach().to(x)"), encoding="utf-8")
        compatibility = "Place MCT mapping coordinates on the input device and dtype."
    fetch(f"{raw}/{source['license']}", root / "LICENSE")
    weights = ROOT / "weights" / source["weights"]
    weights.parent.mkdir(parents=True, exist_ok=True)
    reusable_weights = False
    try:
        previous = json.loads((root / "source.json").read_text(encoding="utf-8"))
        reusable_weights = digest(weights) == previous["weights_sha256"]
    except (OSError, ValueError, KeyError):
        pass
    if not reusable_weights:
        fetch(source["url"], weights)
    write_json(root / "source.json", dict(repository=source["repository"], reference=source["commit"],
        commit=source["commit"] if len(source["commit"]) == 40 else "unknown",
        source_url=f"{raw}/{source['source']}", original_sha256=original_sha256,
        architecture_sha256=digest(architecture), compatibility=compatibility,
        weights_url=source["url"],
        weights_sha256=digest(weights)))
    print(name, weights.stat().st_size, digest(weights), flush=True)


def assets_ready(name):
    """Reuse complete local assets, checking hashes when source metadata supplies them."""
    try:
        if name == "restormer":
            root = ROOT / "third_party/Restormer"
            paths = [root / "basicsr/models/archs/restormer_arch.py", root / "LICENSE.md",
                     root / "source.json", ROOT / "weights/yolo11n.pt"]
            metadata = json.loads((ROOT / "weights/sources.json").read_text(encoding="utf-8"))
            weights = ROOT / "weights/gaussian_color_denoising_blind.pth"
            return all(p.is_file() and p.stat().st_size for p in paths) and digest(weights) == metadata["sha256"]
        source = SOURCES[name]
        root = ROOT / "third_party" / source["directory"]
        metadata = json.loads((root / "source.json").read_text(encoding="utf-8"))
        return ((root / "LICENSE").is_file()
                and digest(root / source["local"]) == metadata["architecture_sha256"]
                and digest(ROOT / "weights" / source["weights"]) == metadata["weights_sha256"])
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["all", "restormer", *SOURCES], default="all")
    parser.add_argument("--check", action="store_true", help="Check local assets without downloading")
    args = parser.parse_args()
    for name in ["restormer", *SOURCES]:
        if args.tool in ("all", name):
            if assets_ready(name):
                print(name, "local assets OK", flush=True)
            elif args.check:
                raise SystemExit(name + ": missing or inconsistent assets; run scripts/setup_assets.py")
            else:
                if name == "restormer":
                    restormer()
                else:
                    image_tool(name)
                if not assets_ready(name):
                    raise SystemExit(name + ": asset check failed")


if __name__ == "__main__": main()
