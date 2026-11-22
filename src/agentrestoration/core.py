from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import traceback
import uuid
from copy import deepcopy
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import truststore
truststore.inject_into_ssl()

_source_root = Path(__file__).resolve().parents[2]
ROOT = Path(os.getenv("AGENTRESTORATION_HOME", str(_source_root if (_source_root / "pyproject.toml").exists() else Path.cwd()))).resolve()
for name in ("TEMP", "TMP", "PIP_CACHE_DIR", "TORCH_HOME", "HF_HOME", "YOLO_CONFIG_DIR"):
    directory = ROOT / ".cache" / name.lower()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def error_details(error):
    """Record validation fields and stack locations without input values or locals."""
    details = dict(type=type(error).__name__, message=type(error).__name__,
                   traceback=[dict(file=frame.filename, line=frame.lineno, function=frame.name)
                              for frame in traceback.extract_tb(error.__traceback__)])
    if isinstance(error, ValidationError):
        details["validation"] = [dict(field=".".join(map(str, item["loc"])) or "configuration",
                                      type=item["type"], message=item["msg"])
                                 for item in error.errors(include_input=False, include_context=False, include_url=False)]
        details["message"] = "数据校验失败：" + "；".join(
            item["field"] + "（" + item["type"] + "）" for item in details["validation"])
    return details


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vlm_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    image_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    vlm_model: str = "qwen3-vl-plus"
    image_model: str = "qwen-image-2.1-turbo"
    ratio_pass: float = Field(default=.01, gt=0)
    ratio_severe: float = .05
    confidence: float = Field(default=.5, ge=0, le=1)
    min_area: float = Field(default=.01, ge=0, le=1)
    position_pass: float = Field(default=.04, gt=0)
    position_severe: float = .12
    match_distance: float = .3
    max_objects: int = Field(default=20, gt=0)
    count_reliability_threshold: float = Field(default=.9, ge=0, le=1)
    threshold_status: str = "uncalibrated"
    timeout: float = Field(default=180, gt=0)
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    image_api_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    tool_weights: str = "weights/gaussian_color_denoising_blind.pth"
    tool_arch: str = "third_party/Restormer/basicsr/models/archs/restormer_arch.py"
    retinex_weights: str = "weights/retinexformer_lol_v1.pth"
    retinex_arch: str = "third_party/Retinexformer/RetinexFormer_arch.py"
    dehaze_weights: str = "weights/dehazeformer_mct_mixed.pth"
    dehaze_arch: str = "third_party/DehazeFormer/dehazeformer.py"
    device: str = "cpu"
    tool_timeout: float = Field(default=300, gt=0)
    restoration_task: Literal["auto", "gaussian_denoising", "low_light", "dehazing"] = "auto"

    @model_validator(mode="after")
    def thresholds(self):
        if self.ratio_severe < self.ratio_pass or self.position_severe < self.position_pass:
            raise ValueError("严重阈值不得低于通过阈值")
        if self.match_distance < self.position_severe:
            raise ValueError("匹配距离不得低于严重位移阈值")
        return self

    @classmethod
    def load(cls, path=None):
        data = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        for key in ("vlm_base_url", "image_base_url", "vlm_model", "image_model"):
            if os.getenv(key.upper()):
                data[key] = os.environ[key.upper()]
        return cls(**data)


class Degradation(BaseModel):
    type: Literal["noise", "blur", "low_light", "haze", "other"]
    severity: Literal["mild", "moderate", "severe", "uncertain"]
    uncertain: bool


class Analysis(BaseModel):
    degradations: list[Degradation]
    restoration_goal: str
    preserve: list[str]
    edit_prompt: str = Field(min_length=1)


class Verdict(BaseModel):
    decision: Literal["accept", "reject", "uncertain"]
    quality_improved: bool | None
    structure_preserved: bool | None
    issues: list[str]
    reason: str
    retry_instruction: str
    structure_issue: Literal["count_change", "position_change", "other", "none", "uncertain"] = "uncertain"
    appearance_change: Literal["normal_repair", "user_requested", "unexpected", "uncertain"] = Field(
        default="uncertain",
        description="normal_repair: expected restoration changes without unexplained color blocks; "
        "user_requested: visible appearance change is justified by the ORIGINAL user request; "
        "unexpected: unrequested background replacement, flat color fill or other artifact; uncertain: cannot confirm.")

    def accepted(self):
        return (self.decision == "accept" and self.quality_improved is True and self.structure_preserved is True
                and self.appearance_change in ("normal_repair", "user_requested"))


def appearance_evidence(original, candidate, box=None):
    """Locate large, smooth, newly chromatic regions; evidence, not a quality verdict."""
    import numpy as np
    from PIL import ImageFilter
    with Image.open(original) as image:
        original_size = image.size
        source = image.convert("RGB")
    with Image.open(candidate) as image:
        result = image.convert("RGB")
    if result.size != source.size:
        result = result.resize(source.size, Image.Resampling.LANCZOS)
    box = tuple(box or (0, 0, *source.size))
    source, result = source.crop(box), result.crop(box)
    size = source.size
    source.thumbnail((256,256))
    result = result.resize(source.size, Image.Resampling.LANCZOS)
    a = np.asarray(source.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)/255
    b = np.asarray(result.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)/255
    chroma_a = a/(a.sum(axis=2,keepdims=True)+1e-6)
    chroma_b = b/(b.sum(axis=2,keepdims=True)+1e-6)
    shift = np.linalg.norm(chroma_b-chroma_a,axis=2)
    saturation = (b.max(axis=2)-b.min(axis=2))/(b.max(axis=2)+1e-6)
    gradient = np.zeros(b.shape[:2], dtype=np.float32)
    gradient[:,1:] = np.max(np.abs(b[:,1:]-b[:,:-1]),axis=2)
    gradient[1:] = np.maximum(gradient[1:],np.max(np.abs(b[1:]-b[:-1]),axis=2))
    mask = (shift>.12) & (saturation>.35) & (gradient<.025)
    height,width=mask.shape
    regions=[]
    # Flood fill bounds memory and avoids a new image-processing dependency.
    visited=np.zeros_like(mask)
    for y,x in zip(*np.nonzero(mask)):
        if visited[y,x]: continue
        visited[y,x]=True
        stack=[(int(y),int(x))]
        points=[]
        while stack:
            py,px=stack.pop();points.append((py,px))
            for ny,nx in ((py-1,px),(py+1,px),(py,px-1),(py,px+1)):
                if 0<=ny<height and 0<=nx<width and mask[ny,nx] and not visited[ny,nx]:
                    visited[ny,nx]=True;stack.append((ny,nx))
        if len(points)<max(16,.03*width*height): continue
        ys,xs=np.asarray(points).T
        absolute=[box[0]+int(xs.min())*size[0]/width,box[1]+int(ys.min())*size[1]/height,
                  box[0]+(int(xs.max())+1)*size[0]/width,box[1]+(int(ys.max())+1)*size[1]/height]
        regions.append(dict(box=[round(v/original_size[i%2],6) for i,v in enumerate(absolute)],
                            area_fraction=round(len(points)/(width*height),6),
                            before_rgb=(a[ys,xs].mean(axis=0)*255).round().astype(int).tolist(),
                            after_rgb=(b[ys,xs].mean(axis=0)*255).round().astype(int).tolist()))
    regions.sort(key=lambda r:r["area_fraction"],reverse=True)
    return dict(status="suspect" if regions else "clear", regions=regions[:8],
                coordinate_space="normalized_full_image", scope="selection" if box!=(0,0,*original_size) else "full_image",
                method="smooth_novel_chroma_v1",
                thresholds=dict(chroma_shift=.12,saturation=.35,gradient=.025,min_area_fraction=.03),
                threshold_status="initial_regression_checked")


def resolve_verdict(verdict, evidence):
    """Resolve contradictions locally without another VLM call."""
    update={}
    if verdict.appearance_change == "unexpected":
        update["decision"]="reject"
    elif verdict.decision == "accept" and (
        verdict.appearance_change == "uncertain" or
        (evidence["status"] == "suspect" and verdict.appearance_change == "normal_repair")
    ):
        update=dict(decision="uncertain",
                    reason="外观变化未确认，或色块证据与正常修复结论冲突。 "+verdict.reason,
                    retry_instruction="Review the reported color regions. Preserve background and appearance unless the original user requested their change.")
    return verdict.model_copy(update=update)


class Job:
    def __init__(self, output, settings, on_event=None, *, log_root=None):
        self.on_event = on_event
        self.path = Path(output) / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:10])
        self.path.mkdir(parents=True)
        self.log_path = self.path if log_root is None else Path(log_root) / self.path.name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.record_path = self.log_path / ("record.json" if log_root is None else "record.log")
        self.events_path = self.log_path / ("events.jsonl" if log_root is None else "events.log")
        self.data = dict(status="preparing", accepted=False, image_requests_sent=0,
                         tool_pipeline_started=0, candidates=[], events=[], steps=[], edit_requests=[], config=settings.model_dump(),
                         created_at=datetime.now(timezone.utc).isoformat(), provenance="real",
                         artifact_directory=str(self.path.resolve()), record_file=str(self.record_path.resolve()),
                         events_file=str(self.events_path.resolve()))
        packages = {}
        for package in ("torch", "ultralytics", "pydantic", "Pillow", "httpx"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "unknown"
        self.data["runtime_packages"] = packages
        weights = ROOT / "weights/yolo11n.pt"
        self.data["detector_weights_sha256"] = digest(weights) if weights.exists() else None
        self.data["implementation_sha256"] = {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}
        self.save()

    def step(self, node_id, title, status, **details):
        node = next((s for s in self.data["steps"] if s["id"] == node_id), None)
        if node is None:
            node = dict(id=node_id, title=title)
            self.data["steps"].append(node)
        node.update(status=status, **details)
        self.data["current_stage"] = title
        self.event("progress", node_id=node_id, title=title, status=status, **details)

    def save(self):
        write_json(self.record_path, self.data)

    def event(self, stage, **details):
        event = dict(time=datetime.now(timezone.utc).isoformat(), stage=stage, **details)
        self.data["events"].append(event)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.save()

        if self.on_event:
            try:
                self.on_event(deepcopy(event), dict(deepcopy(self.data), directory=str(self.path)))
            except Exception as error:
                self.data.setdefault("observer_errors", []).append(type(error).__name__)
                self.save()

    def reserve(self, source):
        field, limit = ("image_requests_sent", 2) if source == "image" else ("tool_pipeline_started", 1)
        if self.data[field] >= limit:
            raise RuntimeError("调用额度耗尽")
        self.data[field] += 1
        self.event("budget_reserved", source=source, count=self.data[field])


def prepare(path, job, settings):
    path = Path(path)
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("仅支持 JPEG、PNG")
    if path.stat().st_size > settings.max_bytes:
        raise ValueError("输入文件超过大小限制")
    shutil.copyfile(path, job.path / ("original_upload" + path.suffix.lower()))
    with Image.open(path) as im:
        if im.format not in ("JPEG", "PNG") or im.width * im.height > settings.max_pixels:
            raise ValueError("图片格式或像素数无效")
        im = ImageOps.exif_transpose(im).convert("RGB")
        dest = job.path / "original_normalized.png"
        im.save(dest)
    job.data["input"] = dict(sha256=digest(path), size=list(im.size), name=path.name)
    job.save()
    return dest


def structure(original, candidate, detector, s, baseline_uncertain=False):
    with Image.open(original) as a, Image.open(candidate) as b:
        error = abs((b.width * a.height) / (b.height * a.width) - 1)
    # The tolerance only removes floating-point roundoff at the exact boundary.
    if error >= s.ratio_pass - 1e-12:
        return dict(status="fail", severe=error >= s.ratio_severe - 1e-12,
                    reason="aspect_ratio", value=error, threshold=s.ratio_pass)
    a, b = detector.detect(original), detector.detect(candidate)
    a = [o for o in a if o["confidence"] >= s.confidence and o["area"] >= s.min_area]
    b = [o for o in b if o["confidence"] >= s.confidence and o["area"] >= s.min_area]
    base = dict(ratio_error=error, original=a, candidate=b, matches=[])
    if not a or max(len(a), len(b)) > s.max_objects:
        return dict(base, status="uncertain", severe=False, reason="unreliable_or_truncated_detection")
    pairs = sorted((math.dist(x["center"], y["center"]), i, j)
                   for i, x in enumerate(a) for j, y in enumerate(b) if x["class"] == y["class"])
    used_a, used_b = set(), set()
    for distance, i, j in pairs:
        if i in used_a or j in used_b or distance > s.match_distance:
            continue
        used_a.add(i)
        used_b.add(j)
        base["matches"].append(dict(original=i, candidate=j, distance=distance))
    moved = [p for p in base["matches"] if p["distance"] > s.position_pass]
    if moved:
        return dict(base, status="fail", severe=any(p["distance"] >= s.position_severe for p in moved),
                    reason="object_position", threshold=s.position_pass)
    # Confidence alone cannot establish whether an unmatched detection is a real deletion.
    if len(used_a) != len(a) or len(used_b) != len(b):
        unmatched = [x for i,x in enumerate(a) if i not in used_a] + [x for i,x in enumerate(b) if i not in used_b]
        reliable = (not baseline_uncertain and bool(base["matches"]) and
            all(x["confidence"] >= s.count_reliability_threshold for x in a + unmatched))
        if reliable:
            return dict(base, status="fail", severe=True, reason="high_confidence_suspected_count_change",
                        confidence_threshold=s.count_reliability_threshold,
                        unmatched_original=sorted(set(range(len(a))) - used_a),
                        unmatched_candidate=sorted(set(range(len(b))) - used_b))
        return dict(base, status="uncertain", severe=False, reason="suspected_count_change",
                    unmatched_original=sorted(set(range(len(a))) - used_a),
                    unmatched_candidate=sorted(set(range(len(b))) - used_b))
    return dict(base, status="pass", severe=False, reason="matched")


def choose_route(severe, available, images, tools):
    if severe and available and tools < 1:
        return "tool"
    if images < 2:
        return "image"
    if available and tools < 1:
        return "tool"
    return "stop"
