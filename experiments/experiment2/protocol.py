"""Frozen sample selection and full-reference evaluation rules."""
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from agentrestoration.core import digest


def pixels(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def evaluate(output, reference, destination, ratio_limit=.01):
    candidate, target = pixels(output), pixels(reference)
    ratio_error = abs(candidate.width*target.height/(candidate.height*target.width)-1)
    result = dict(original_size=list(candidate.size), reference_size=list(target.size),
                  ratio_error=ratio_error, eligible=ratio_error < ratio_limit-1e-12,
                  mapping="none", psnr=None, ssim=None)
    if not result["eligible"]:
        return dict(result, reason="aspect_ratio_failed")
    if candidate.size != target.size:
        candidate = candidate.resize(target.size, Image.Resampling.LANCZOS)
        result["mapping"] = "LANCZOS_to_reference_size_after_ratio_gate"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(destination)
    a, b = np.asarray(target), np.asarray(candidate)
    mse = float(np.mean((a.astype(float)-b.astype(float))**2))
    # Exact matches have infinite PSNR; JSON represents them explicitly, not as NaN/Infinity.
    psnr = float(peak_signal_noise_ratio(a,b,data_range=255)) if mse else None
    return dict(result, psnr=psnr, perfect_match=mse == 0,
                ssim=float(structural_similarity(a,b,data_range=255,channel_axis=2,
                                                 win_size=7,gaussian_weights=False,
                                                 use_sample_covariance=True)),
                evaluated_image=str(destination.resolve()))


def make_manifest(data_root, count, seed, output):
    base = Path(data_root)
    paths = sorted((base/"CBSD68_25").glob("*.png"))
    pairs = [(p,base/"CBSD68"/p.name) for p in paths if (base/"CBSD68"/p.name).is_file()]
    random.Random(seed).shuffle(pairs)
    if len(pairs) < count:
        raise ValueError("Not enough paired scenes")
    rows = []
    for source, reference in pairs[:count]:
        a, b = pixels(source), pixels(reference)
        if a.size != b.size:
            raise ValueError("Pair dimensions differ: "+source.name)
        residual = np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
        low = np.asarray(a.filter(ImageFilter.GaussianBlur(2)),dtype=float)-np.asarray(
            b.filter(ImageFilter.GaussianBlur(2)),dtype=float)
        check = dict(residual_rmse=float(np.sqrt(np.mean(residual**2))),
                     residual_mean=float(residual.mean()), lowpass_rmse=float(np.sqrt(np.mean(low**2))))
        if not (7.65 < check["residual_rmse"] < 38.25 and
                abs(check["residual_mean"]) < 5 and check["lowpass_rmse"] < 8):
            raise ValueError("Pair numerical consistency failed: "+source.name)
        baseline = evaluate(source,reference,Path(output)/"input_metrics"/source.name)
        rows.append(dict(scene=source.stem, input=str(source.resolve()), reference=str(reference.resolve()),
                         input_sha256=digest(source), reference_sha256=digest(reference),
                         dimensions=list(a.size), pairing=check, input_metrics=baseline))
    return rows
