"""Download upstream assets into this workspace; no credentials involved."""
import json
import argparse
import time
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentrestoration.core import ROOT, digest, write_json

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(url,path)
    print(path.name, digest(path), flush=True)

def restormer():
    sha=json.load(urllib.request.urlopen("https://api.github.com/repos/swz30/Restormer/commits/main"))["sha"]
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
    if not weights.exists():
        fetch(source["url"], weights)
    write_json(root / "source.json", dict(repository=source["repository"], reference=source["commit"],
        commit=source["commit"] if len(source["commit"]) == 40 else "unknown",
        source_url=f"{raw}/{source['source']}", original_sha256=original_sha256,
        architecture_sha256=digest(architecture), compatibility=compatibility,
        weights_url=source["url"],
        weights_sha256=digest(weights)))
    print(name, weights.stat().st_size, digest(weights), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["all", "restormer", *SOURCES], default="all")
    args = parser.parse_args()
    if args.tool in ("all", "restormer"):
        restormer()
    for name in SOURCES:
        if args.tool in ("all", name):
            image_tool(name)


if __name__ == "__main__": main()
