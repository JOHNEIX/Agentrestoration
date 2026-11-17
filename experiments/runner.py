"""Deterministic sample manifest, calibration and A/B/C/D experiments."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentrestoration.core import ROOT, Settings, digest, write_json

DIRECTORY=ROOT / "experiments"

def manifest():
    from PIL import Image
    base=Path(r"F:\cvlab\datasets\test-data\denoising_testsets")
    pairs=[(p,base / "CBSD68" / p.name) for p in sorted((base / "CBSD68_25").glob("*.png"))]
    pairs=[(a,b) for a,b in pairs if b.exists()]
    random.Random(20261112).shuffle(pairs)
    if len(pairs)<20: raise RuntimeError("Insufficient paired scenes")
    rows=[]
    for index,(a,b) in enumerate(pairs[:20]):
        with Image.open(a) as x,Image.open(b) as y:
            if x.size != y.size: raise RuntimeError("Pair size mismatch")
        rows.append(dict(scene=a.stem,split="calibration" if index<8 else "test",input=str(a),reference=str(b),
                         input_sha256=digest(a),reference_sha256=digest(b),pairing="filename_and_dimensions_only; visually_verify"))
    write_json(DIRECTORY / "manifest.json",rows)
    print("Manifest: 8 calibration / 12 test")

def calibrate():
    import numpy as np
    from PIL import Image, ImageEnhance
    from agentrestoration.adapters import Detector
    from agentrestoration.core import structure
    settings=Settings.load(ROOT / "configs/default.json")
    settings.restoration_task="gaussian_denoising"
    detector=Detector(settings)
    rows=json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    distances=[]; controls=[]
    for row in rows:
        if row["split"]!="calibration": continue
        dest=DIRECTORY / "calibration" / (row["scene"]+"_brightness.png")
        dest.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(row["input"]) as im:
            ImageEnhance.Brightness(im).enhance(1.1).save(dest)
        check=structure(row["input"],dest,detector,settings)
        controls.append(dict(scene=row["scene"],check=check))
        distances.extend(m["distance"] for m in check.get("matches",[]))
    # Calibration only measures detector stability under content-preserving brightness.
    # Too little evidence is explicitly not promoted to calibrated status.
    if len(distances)<3:
        write_json(DIRECTORY / "calibration.json",dict(status="insufficient_evidence",controls=controls))
        raise RuntimeError("Insufficient reliable matched detections for threshold freezing")
    position=max(.02,min(.08,float(np.quantile(distances,.95))+.01))
    settings.position_pass=position
    settings.position_severe=max(.12,3*position)
    settings.match_distance=max(.3,settings.position_severe)
    settings.threshold_status="pilot_calibrated_limited_brightness_controls"
    write_json(DIRECTORY / "frozen_config.json",settings.model_dump())
    write_json(DIRECTORY / "calibration.json",dict(status=settings.threshold_status,matched_objects=len(distances),
        caveat="Confidence/area remain engineering defaults; synthetic stability does not prove real edit reliability.",controls=controls))

def metrics(path,reference):
    import numpy as np
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio,structural_similarity
    with Image.open(path) as a,Image.open(reference) as b:
        x,y=np.asarray(a.convert("RGB")),np.asarray(b.convert("RGB"))
    if x.shape!=y.shape: return dict(psnr=None,ssim=None,reason="dimensions_differ")
    return dict(psnr=float(peak_signal_noise_ratio(y,x,data_range=255)),
                ssim=float(structural_similarity(y,x,data_range=255,channel_axis=2)))

def experiments():
    from agentrestoration.adapters import Qwen,Detector,Restormer
    from experiments.pipeline import run
    from agentrestoration.core import structure,Job
    s=Settings.load(DIRECTORY / "frozen_config.json")
    s.restoration_task="gaussian_denoising"
    api,detector,tool=Qwen(s),Detector(s),Restormer(s)
    manifest=json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    results=[]
    for row in manifest:
        if row["split"]!="test": continue
        for group in "ABCD":
            job=run(row["input"],"去除噪声，保留原始内容与构图",s,api,detector,tool,DIRECTORY / "runs",group)
            if job.data["status"]=="execution_failed":
                write_json(DIRECTORY / "results.partial.json",results)
                raise RuntimeError("Execution failure: "+str(job.path))
            output=job.path / job.data["output"]
            checks=structure(row["input"],output,detector,s)
            offline=api.compare(row["input"],output,checks,job)
            result=dict(scene=row["scene"],group=group,status=job.data["status"],accepted=job.data["accepted"],
                images=job.data["image_requests_sent"],tools=job.data["tool_pipeline_started"],
                seconds=job.data["elapsed_seconds"],directory=str(job.path),offline_verdict=offline.model_dump(),
                metrics=metrics(output,row["reference"]),input_metrics=metrics(row["input"],row["reference"]),
                human_review="pending")
            results.append(result)
            write_json(DIRECTORY / "results.json",results)
            print(row["scene"],group,job.data["status"],flush=True)
    with (DIRECTORY / "human_review.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f); writer.writerow(["scene","group","directory","quality","large_scale_error","false_accept","false_reject","notes"])
        for r in results: writer.writerow([r["scene"],r["group"],r["directory"],"pending","pending","pending","pending",""])

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=["manifest","calibrate","run"])
    action=parser.parse_args().action
    {"manifest":manifest,"calibrate":calibrate,"run":experiments}[action]()
