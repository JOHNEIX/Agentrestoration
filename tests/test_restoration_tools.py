"""Routing and pixel contracts; model quality is checked with author weights separately."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agentrestoration.core import Analysis, Settings
from agentrestoration.restoration_tools import RestorationTools, Retinexformer, DehazeFormer


def analysis(*types):
    return Analysis(degradations=[dict(type=t, severity=s, uncertain=u) for t, s, u in types],
                    restoration_goal="restore", preserve=[], edit_prompt="restore")


def ready_settings(tmp_path, **overrides):
    asset = tmp_path / "asset"
    asset.touch()
    return Settings(retinex_weights=str(asset), retinex_arch=str(asset),
                    dehaze_weights=str(asset), dehaze_arch=str(asset), **overrides)


@pytest.mark.parametrize("task,kind,name", [
    ("auto", "low_light", "Retinexformer"), ("low_light", "low_light", "Retinexformer"),
    ("auto", "haze", "DehazeFormer"), ("dehazing", "haze", "DehazeFormer")])
def test_matching_tools(tmp_path, task, kind, name):
    tools = RestorationTools(ready_settings(tmp_path, restoration_task=task))
    assert tools.available(analysis((kind, "moderate", False)))
    assert tools.name == name


@pytest.mark.parametrize("task,kind,uncertain", [
    ("auto", "noise", False), ("auto", "blur", False), ("low_light", "haze", False),
    ("dehazing", "low_light", False), ("auto", "low_light", True), ("auto", "haze", True)])
def test_mismatched_or_uncertain_tools(tmp_path, task, kind, uncertain):
    tools = RestorationTools(ready_settings(tmp_path, restoration_task=task))
    assert not tools.available(analysis((kind, "severe", uncertain)))
    assert tools.selected is None


def test_missing_assets_and_severity_selection(tmp_path):
    tools = RestorationTools(ready_settings(tmp_path))
    assert tools.available(analysis(("low_light", "mild", False), ("haze", "severe", False)))
    assert tools.name == "DehazeFormer"
    assert tools.available(analysis(("low_light", "severe", False), ("haze", "severe", False)))
    assert tools.name == "Retinexformer"
    tools.tools[1].s.retinex_weights = str(tmp_path / "missing")
    assert tools.available(analysis(("low_light", "severe", False), ("haze", "mild", False)))
    assert tools.name == "DehazeFormer"
    assert not tools.available(analysis(("low_light", "severe", False)))
    assert tools.selected is None  # previous selection must not leak into a new analysis
    assert not tools.available(analysis(("low_light", "uncertain", False)))


@pytest.mark.parametrize("kind,name", [("low_light", "Retinexformer"), ("haze", "DehazeFormer")])
def test_tool_candidate_is_verified_and_named(tmp_path, monkeypatch, kind, name):
    from test_core import FakeAPI, Detection
    from agentrestoration.pipeline import run
    tools = RestorationTools(ready_settings(tmp_path))
    class API(FakeAPI):
        def analyze(self, *args): return analysis((kind, "moderate", False))
        def compare(self, original, candidate, *args):
            self.accept = candidate.name == "tool_1.png"
            return super().compare(original, candidate, *args)
    def restore(original, job):
        with Image.open(original) as image:
            image.save(job.path / "tool_1.png")
        return job.path / "tool_1.png"
    for tool in tools.tools[1:]:
        monkeypatch.setattr(tool, "run", restore)
    source = tmp_path / "input.png"
    Image.new("RGB", (100,100)).save(source)
    job = run(source, "restore", tools.tools[0].s, API(), Detection(), tools,
              tmp_path / "runs", log_root=tmp_path / "log/tasks")
    assert job.data["status"] == "completed"
    assert job.data["image_requests_sent"] == 2 and job.data["tool_pipeline_started"] == 1
    candidate = job.data["candidates"][-1]
    assert candidate["tool_name"] == name
    assert name in candidate["label"]
    assert (job.path / "final.png").is_file()


@pytest.mark.parametrize("tool_class", [Retinexformer, DehazeFormer])
@pytest.mark.parametrize("size", [(1, 1), (37, 19), (289, 65)])
def test_inference_range_padding_and_overlap(tmp_path, monkeypatch, tool_class, size):
    import torch
    architecture = tmp_path / "architecture.py"
    architecture.write_text("# stand-in model for pixel-contract tests\n")
    checkpoint = tmp_path / "weights.pth"
    torch.save({}, checkpoint)
    kwargs = {"retinex_arch": str(architecture), "retinex_weights": str(checkpoint)} if tool_class is Retinexformer else {
        "dehaze_arch": str(architecture), "dehaze_weights": str(checkpoint)}
    tool = tool_class(Settings(**kwargs))
    seen = []
    class Identity(torch.nn.Module):
        def forward(self, pixels):
            seen.append(pixels.detach().clone())
            if tool.tiled:
                assert pixels.shape[-1] % 32 == pixels.shape[-2] % 32 == 0
            return pixels
    monkeypatch.setattr(tool, "build", lambda module: Identity())
    source = tmp_path / "input.png"
    pixels = np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    events = []
    job = SimpleNamespace(path=tmp_path, event=lambda stage, **data: events.append((stage, data)))
    result = tool.infer(source, job)
    np.testing.assert_array_equal(np.asarray(Image.open(result)), pixels)
    expected = torch.from_numpy(pixels).permute(2, 0, 1).float()[None] / 255
    if tool_class is DehazeFormer:
        expected = expected * 2 - 1
    first_width = min(size[0], 288) if tool.tiled else size[0]
    torch.testing.assert_close(seen[0][:, :, :size[1], :first_width], expected[:, :, :, :first_width])
    assert events[0][1]["name"] == tool.name


def test_real_router_tool_failure_keeps_budget_and_retries(tmp_path, monkeypatch):
    from test_core import FakeAPI, Detection
    from agentrestoration.pipeline import run
    class API(FakeAPI):
        def analyze(self, *args): return analysis(("low_light", "severe", False))
        def edit(self, original, prompt, job):
            result = super().edit(original, prompt, job)
            if job.data["image_requests_sent"] == 1:
                Image.new("RGB", (120, 100)).save(result)
            return result
    tools = RestorationTools(ready_settings(tmp_path))
    calls = []
    def fail(original, job):
        calls.append("Retinexformer")
        raise RuntimeError("simulated tool failure")
    monkeypatch.setattr(tools.tools[1], "run", fail)
    source = tmp_path / "input.png"
    Image.new("RGB", (100, 100)).save(source)
    job = run(source, "restore", tools.tools[0].s, API(accept=True), Detection(), tools,
              tmp_path / "runs", log_root=tmp_path / "log/tasks")
    assert calls == ["Retinexformer"]
    assert job.data["image_requests_sent"] == 2
    assert job.data["tool_pipeline_started"] == 1
    assert job.data["status"] == "completed"
