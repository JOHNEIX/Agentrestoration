"""Task-specific restoration tools sharing the isolated inference worker."""
import importlib.util

from .adapters import APIError, Restormer
from .core import ROOT, Analysis, digest


class ImageTool(Restormer):
    task = None
    degradation = None
    tiled = True

    def available(self, analysis):
        return (self.s.restoration_task in ("auto", self.task)
                and (ROOT / self.weights).is_file() and (ROOT / self.arch).is_file()
                and any(d.type == self.degradation and not d.uncertain and d.severity != "uncertain"
                        for d in analysis.degradations))

    def infer(self, original, job):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        torch.set_num_threads(4)
        architecture = ROOT / self.arch
        spec = importlib.util.spec_from_file_location(self.worker_name + "_arch", architecture)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = self.build(module).to(self.s.device)
        checkpoint = torch.load(ROOT / self.weights, map_location="cpu", weights_only=True)
        state = checkpoint.get(self.state_key, checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        job.event("tool_version", name=self.name, variant=self.variant, task=self.task,
                  weights_sha256=digest(ROOT / self.weights), architecture_sha256=digest(architecture),
                  torch_version=str(torch.__version__), device=self.s.device,
                  tile_stride=256 if self.tiled else None, tile_halo=32 if self.tiled else 0,
                  blending="overlap_average" if self.tiled else "full_image_curve_mapping", input_range=self.input_range)
        with Image.open(original) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        x = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.s.device)
        height, width = x.shape[-2:]
        stride, halo = (256, 32) if self.tiled else (max(height, width), 0)
        output, count = torch.zeros_like(x), torch.zeros_like(x)
        with torch.inference_mode():
            for top in range(0, height, stride):
                for left in range(0, width, stride):
                    t, l = max(0, top - halo), max(0, left - halo)
                    bottom, right = min(height, top + stride + halo), min(width, left + stride + halo)
                    tile = x[:, :, t:bottom, l:right]
                    th, tw = tile.shape[-2:]
                    padded_h, padded_w = max(32, (th + 31)//32*32), max(32, (tw + 31)//32*32)
                    if self.tiled:
                        tile = F.pad(tile, (0, padded_w-tw, 0, padded_h-th), mode="replicate")
                    if self.input_range == "minus_one_to_one":
                        tile = tile * 2 - 1
                    restored = model(tile)
                    if self.input_range == "minus_one_to_one":
                        restored = restored.clamp(-1, 1) * .5 + .5
                    if not torch.isfinite(restored).all():
                        raise APIError("nonfinite_tool_output")
                    output[:, :, t:bottom, l:right] += restored[:, :, :th, :tw]
                    count[:, :, t:bottom, l:right] += 1
        pixels = ((output/count).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255).round().astype("uint8")
        destination = job.path / "tool_1.png"
        Image.fromarray(pixels).save(destination)
        return destination


class Retinexformer(ImageTool):
    name, worker_name = "Retinexformer", "retinexformer"
    task, degradation = "low_light", "low_light"
    variant, state_key, input_range = "LOL-v1", "params", "zero_to_one"

    @property
    def weights(self): return self.s.retinex_weights

    @property
    def arch(self): return self.s.retinex_arch

    def build(self, module):
        return module.RetinexFormer(stage=1, n_feat=40, num_blocks=[1, 2, 2])


class DehazeFormer(ImageTool):
    name, worker_name = "DehazeFormer", "dehazeformer"
    task, degradation = "dehazing", "haze"
    variant, state_key, input_range = "MCT-mixed", "state_dict", "minus_one_to_one"
    tiled = False

    @property
    def weights(self): return self.s.dehaze_weights

    @property
    def arch(self): return self.s.dehaze_arch

    def build(self, module):
        return module.MCT()


class RestorationTools:
    def __init__(self, settings):
        self.tools = [Restormer(settings), Retinexformer(settings), DehazeFormer(settings)]
        self.selected = None

    @property
    def name(self): return self.selected.name if self.selected else "专用工具"

    def available(self, analysis):
        # For mixed degradation, use the most severe eligible task; ties retain analysis order.
        severity = {"severe": 3, "moderate": 2, "mild": 1, "uncertain": 0}
        ordered = sorted(enumerate(analysis.degradations), key=lambda item: (-severity[item[1].severity], item[0]))
        for _, degradation in ordered:
            if degradation.uncertain or degradation.severity == "uncertain":
                continue
            single = Analysis(degradations=[degradation], restoration_goal=analysis.restoration_goal,
                              preserve=analysis.preserve, edit_prompt=analysis.edit_prompt)
            for tool in self.tools:
                if tool.available(single):
                    self.selected = tool
                    return True
        self.selected = None
        return False

    def run(self, original, job):
        if self.selected is None:
            raise APIError("no_applicable_tool")
        job.event("tool_selected", name=self.selected.name)
        return self.selected.run(original, job)


WORKERS = {"restormer": Restormer, "retinexformer": Retinexformer, "dehazeformer": DehazeFormer}
