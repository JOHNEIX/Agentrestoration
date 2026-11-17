"""Download upstream assets into this workspace; no credentials involved."""
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentrestoration.core import ROOT, digest, write_json

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(url,path)
    print(path.name, digest(path), flush=True)

def main():
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

if __name__ == "__main__": main()
