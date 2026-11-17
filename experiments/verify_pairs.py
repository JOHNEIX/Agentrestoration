"""Check low-frequency spatial agreement and residuals without altering splits."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from agentrestoration.core import ROOT,write_json

rows=json.loads((ROOT/"experiments/manifest.json").read_text(encoding="utf-8"))
checks=[]
for row in rows:
    with Image.open(row["input"]) as a,Image.open(row["reference"]) as b:
        a,b=a.convert("RGB"),b.convert("RGB")
        x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if x.shape!=y.shape:raise RuntimeError("shape mismatch")
        residual=x-y
        low=np.asarray(a.filter(ImageFilter.GaussianBlur(2)),dtype=float)-np.asarray(b.filter(ImageFilter.GaussianBlur(2)),dtype=float)
        check=dict(scene=row["scene"],residual_rmse=float(np.sqrt(np.mean(residual**2))),
                   residual_mean=float(np.mean(residual)),lowpass_rmse=float(np.sqrt(np.mean(low**2))))
        check["numerically_consistent"]=(7.65<check["residual_rmse"]<38.25 and abs(check["residual_mean"])<5 and check["lowpass_rmse"]<8)
        checks.append(check)
write_json(ROOT/"experiments/pairing_checks.json",dict(checks=checks,
    caveat="Numerical consistency supports same-scene pairing; heuristic checks are not a substitute for human content labels."))
print(f"Numerically consistent pairs: {sum(c['numerically_consistent'] for c in checks)}/{len(checks)}")
