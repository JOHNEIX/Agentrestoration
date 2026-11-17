import base64
import importlib.util
import json
import os
import re
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .core import Analysis, ROOT, Verdict, digest


class APIError(RuntimeError):
    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def safe_error_text(value, keys):
    text = str(value or "")
    for key in keys:
        if key:
            text = text.replace(key, "[REDACTED]")
    text = re.sub(r"data:[^,\s]+;base64,[A-Za-z0-9+/=]+", "[IMAGE_DATA]", text)
    text = re.sub(r"https?://[^\s\"<>]+", "[URL]", text)
    text = re.sub(r"(?i)Bearer\s+[^\s\"<>]+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_.-]+", "[REDACTED]", text)
    return text[:1200]


def credentials():
    keys = {}
    path = ROOT / "api.txt"
    if path.exists():
        current = None
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "识别" in line:
                current = "vlm"
            elif "编辑" in line:
                current = "image"
            elif line.strip() and current:
                keys[current] = line.strip()
                current = None
    return {n: os.getenv(n.upper() + "_API_KEY") or keys.get(n, "") for n in ("vlm", "image")}


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


class Qwen:
    def __init__(self, settings):
        self.s = settings
        self.keys = credentials()
        self.client = httpx.Client(timeout=settings.timeout, transport=httpx.HTTPTransport(retries=0))

    def post(self, kind, endpoint, body, job):
        if not self.keys[kind]:
            raise APIError("缺少 " + kind + " 凭据")
        url = getattr(self.s, kind + "_base_url").rstrip("/") + endpoint
        job.event("api_request", kind=kind, model=body.get("model"))
        try:
            response = self.client.post(url, headers={"Authorization": "Bearer " + self.keys[kind]}, json=body)
        except httpx.HTTPError as e:
            raise APIError("network_" + type(e).__name__) from None
        if response.is_error:
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            error = payload.get("error", payload) if isinstance(payload, dict) else {}
            error = error if isinstance(error, dict) else {}
            code = safe_error_text(error.get("code"), self.keys.values())
            message = safe_error_text(error.get("message"), self.keys.values())
            request_id = safe_error_text(payload.get("request_id", response.headers.get("x-request-id")), self.keys.values()) if isinstance(payload, dict) else None
            job.event("api_error", kind=kind, http_status=response.status_code, provider_code=code,
                      message=message, request_id=request_id)
            detail = " | ".join(v for v in (code, message) if v)
            raise APIError(f"HTTP_{response.status_code}" + (": " + detail if detail else ""), status_code=response.status_code)
        result = response.json()
        job.event("api_response", kind=kind, request_id=result.get("id", result.get("request_id")),
                  model=result.get("model", "unknown"), usage=result.get("usage"))
        return result

    def vlm(self, paths, instruction, schema, job):
        content = [{"type": "text", "text": instruction + "\nReturn a populated JSON INSTANCE, NOT the schema. "
            "Use these exact top-level keys: " + ", ".join(schema.model_fields) + ". Fill each field with your image assessment. "
            "Never output properties, required, title or type as top-level keys. JSON Schema for validation only: " + json.dumps(schema.model_json_schema())}]
        content += [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in paths]
        for attempt in range(2):
            try:
                result = self.post("vlm", "/chat/completions", dict(model=self.s.vlm_model,
                    messages=[dict(role="user", content=content)], response_format={"type": "json_object"},
                    enable_thinking=False), job)
                from pydantic import ValidationError
                try:
                    return schema.model_validate_json(result["choices"][0]["message"]["content"])
                except ValidationError as e:
                    job.event("schema_error", fields=[dict(location=list(x["loc"]),type=x["type"]) for x in e.errors()])
                    raise
            except APIError as e:
                if attempt or not (str(e).startswith("network_") or e.status_code in (429, 500, 502, 503)):
                    raise
                job.event("vlm_network_retry", error=str(e))

    def analyze(self, original, requirement, job):
        return self.vlm([original], "Analyze degradations conservatively. Generate restoration prompt preserving all content and composition. User: " + requirement, Analysis, job)

    def compare(self, original, candidate, checks, job):
        return self.vlm([original, candidate], "Image 1 is original; image 2 is candidate. Compare quality improvement and preserved content. "
            "Reject additions, deletions, movement, redraw or side effects. Accept only clear improvement and preserved structure. "
            "When detection is uncertain explicitly review objects; if uncertain do not accept. Checks: " + json.dumps(checks), Verdict, job)

    def image_input(self, original, job, index):
        raw = Path(original).read_bytes()
        limit = self.s.image_api_max_bytes
        def fits(data):
            return len(data) <= limit and 4 * ((len(data) + 2) // 3) + 32 <= limit
        with Image.open(original) as source:
            original_size = list(source.size)
            mime = Image.MIME.get(source.format, "image/png")
            image = source.convert("RGB") if not fits(raw) else None
        metadata = dict(original_bytes=len(raw), original_size=original_size,
                        sent_bytes=len(raw), sent_size=original_size, mime=mime)
        if image is not None:
            prepared = None
            for scale_attempt in range(8):
                for quality in (95, 90, 85):
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=True)
                    encoded = buffer.getvalue()
                    if fits(encoded):
                        prepared = encoded
                        break
                if prepared is not None:
                    break
                image = image.resize((max(1, int(image.width * .8)), max(1, int(image.height * .8))), Image.Resampling.LANCZOS)
            if prepared is None:
                raise APIError("image_payload_too_large")
            raw, mime = prepared, "image/jpeg"
            folder = job.path / "api_inputs"
            folder.mkdir(exist_ok=True)
            prepared_path = f"api_inputs/image_{index}.jpg"
            (job.path / prepared_path).write_bytes(raw)
            metadata.update(sent_bytes=len(raw), sent_size=list(image.size), mime=mime,
                            quality=quality, path=prepared_path)
        job.event("api_image_prepared", index=index, **metadata)
        return f"data:{mime};base64," + base64.b64encode(raw).decode(), metadata

    def edit(self, original, prompt, job):
        if not self.keys["image"]:
            raise APIError("缺少 image 凭据")
        image, image_metadata = self.image_input(original, job, job.data["image_requests_sent"] + 1)
        body = dict(model=self.s.image_model, prompt=prompt, image=image, n=1)
        job.reserve("image")
        index = job.data["image_requests_sent"]
        prompt_path = f"prompts/image_{index}.txt"
        (job.path / "prompts").mkdir(exist_ok=True)
        (job.path / prompt_path).write_text(prompt, encoding="utf-8")
        request_record = dict(index=index, prompt=prompt, prompt_file=prompt_path,
                              candidate=f"image_{index}.png", status="sending", input=image_metadata)
        job.data["edit_requests"].append(request_record)
        job.event("edit_request_recorded", index=index, prompt_file=prompt_path)
        try:
            result = self.post("image", "/images/generations", body, job)
        except Exception as error:
            request_record.update(status="failed", error=str(error) if isinstance(error, APIError) else type(error).__name__)
            job.save()
            raise
        request_record.update(status="response_received", request_id=result.get("id", result.get("request_id")))
        job.save()
        dest = job.path / f"image_{job.data['image_requests_sent']}.png"
        job.event("candidate_path_reserved", path=dest.name)
        item = result["data"][0]
        for attempt in range(2):
            try:
                with self.client.stream("GET", item["url"]) as r:
                    r.raise_for_status()
                    size = 0
                    with dest.open("wb") as f:
                        for chunk in r.iter_bytes():
                            size += len(chunk)
                            if size > self.s.max_bytes:
                                raise APIError("output_too_large")
                            f.write(chunk)
                break
            except httpx.HTTPError:
                if attempt:
                    raise APIError("download_failed") from None
        with Image.open(dest) as im:
            if im.width * im.height > self.s.max_pixels:
                raise APIError("output_pixels_exceeded")
            im.verify()
        request_record["status"] = "completed"
        job.save()
        return dest


class Detector:
    def __init__(self, settings):
        from ultralytics import YOLO
        self.s = settings
        self.model = YOLO(str(ROOT / "weights" / "yolo11n.pt"))
        self.sha256 = digest(ROOT / "weights" / "yolo11n.pt")

    def detect(self, path):
        result = self.model.predict(str(path), conf=self.s.confidence, device="cpu", verbose=False)[0]
        h, w = result.orig_shape
        objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append(dict(class_id=int(box.cls.item()), **{"class": result.names[int(box.cls.item())]},
                confidence=float(box.conf.item()), center=[(x1+x2)/(2*w), (y1+y2)/(2*h)],
                area=(x2-x1)*(y2-y1)/(w*h), box=[x1/w, y1/h, x2/w, y2/h]))
        return objects


class Restormer:
    def __init__(self, settings):
        self.s = settings

    def available(self, analysis):
        return self.s.restoration_task == "gaussian_denoising" and (ROOT / self.s.tool_weights).is_file() and (ROOT / self.s.tool_arch).is_file() and any(
            d.type == "noise" and not d.uncertain for d in analysis.degradations)

    def run(self, original, job):
        import subprocess
        import sys
        from .core import write_json
        config = job.path / "tool_config.json"
        write_json(config, self.s.model_dump())
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run([sys.executable, "-m", "agentrestoration.tool_worker", str(original.resolve()),
            str(job.path.resolve()), str(config.resolve())], env=env, capture_output=True, timeout=self.s.tool_timeout)
        if result.returncode:
            error_path = job.path / "tool_error.json"
            kind = json.loads(error_path.read_text(encoding="utf-8")).get("kind", "subprocess_failed") if error_path.exists() else "subprocess_failed"
            raise APIError("tool_" + kind)
        metadata = json.loads((job.path / "tool_metadata.json").read_text(encoding="utf-8"))
        job.event("tool_version", **metadata)
        return job.path / "tool_1.png"

    def infer(self, original, job):
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(4)
        arch = ROOT / self.s.tool_arch
        spec = importlib.util.spec_from_file_location("restormer_arch", arch)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Restormer(LayerNorm_type="BiasFree").to(self.s.device)
        state = torch.load(ROOT / self.s.tool_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state["params"])
        model.eval()
        job.event("tool_version", weights_sha256=digest(ROOT / self.s.tool_weights), architecture_sha256=digest(arch),
                  torch_version=str(torch.__version__), device=self.s.device, tile_stride=224, tile_halo=16,
                  layer_norm="BiasFree", blending="overlap_average")
        with Image.open(original) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.s.device)
        h, w = x.shape[-2:]
        # Tile inference bounds attention memory on the 6 GB device.
        output = torch.zeros_like(x)
        count = torch.zeros_like(x)
        with torch.inference_mode():
            for top in range(0, h, 224):
                for left in range(0, w, 224):
                    t, l = max(0, top-16), max(0, left-16)
                    bottom, right = min(h, top+240), min(w, left+240)
                    tile = x[:, :, t:bottom, l:right]
                    th, tw = tile.shape[-2:]
                    tile = F.pad(tile, (0, (-tw)%8, 0, (-th)%8), mode="replicate")
                    output[:, :, t:bottom, l:right] += model(tile)[:, :, :th, :tw]
                    count[:, :, t:bottom, l:right] += 1
        arr = ((output/count).clamp(0, 1)[0].permute(1,2,0).cpu().numpy()*255).round().astype("uint8")
        dest = job.path / "tool_1.png"
        Image.fromarray(arr).save(dest)
        return dest
