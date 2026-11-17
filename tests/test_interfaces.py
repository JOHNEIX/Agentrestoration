import importlib.util
from types import SimpleNamespace

import pytest
from PIL import Image

from agentrestoration.core import ROOT, Settings, Analysis


def experimental_pipeline():
    spec = importlib.util.spec_from_file_location("experimental_pipeline", ROOT / "experiments/pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_system_cli_rejects_experiment_group(monkeypatch, capsys):
    from agentrestoration.cli import main
    monkeypatch.setattr("sys.argv", ["restore", "image.png", "--group", "A"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "unrecognized arguments: --group A" in capsys.readouterr().err


@pytest.mark.parametrize("group", ["C", "D"])
def test_experimental_ablation_calls_full_system(group, monkeypatch):
    experiment = experimental_pipeline()
    tool = object()
    called = []
    options = []
    result = SimpleNamespace(data={}, save=lambda: None)
    def restore(*args, **kwargs):
        called.append(args)
        options.append(kwargs)
        return result
    monkeypatch.setattr(experiment, "restore", restore)
    assert experiment.run("image", "goal", None, None, None, tool, "output", group) is result
    assert len(called[0]) == 7  # system has no experimental group argument
    assert options == [{"log_root": None}]
    selected_tool = called[0][5]
    assert (selected_tool is tool) == (group == "D")
    if group == "C":
        assert not selected_tool.available(None)


@pytest.mark.parametrize("group", ["A", "B"])
def test_experimental_baselines_stay_outside_system(group, tmp_path):
    experiment = experimental_pipeline()
    calls = []
    class API:
        provenance = "simulated"
        def analyze(self, *args):
            calls.append("analyze")
            return Analysis(degradations=[], restoration_goal="goal", preserve=[], edit_prompt="restore")
        def edit(self, original, prompt, job):
            calls.append("edit")
            job.reserve("image")
            output = job.path / "image.png"
            with Image.open(original) as im:
                im.save(output)
            return output
    source = tmp_path / "input.png"
    Image.new("RGB", (64,64)).save(source)
    job = experiment.run(source, "goal", Settings(), API(), None, None, tmp_path / "runs", group)
    assert calls == (["edit"] if group == "A" else ["analyze", "edit"])
    assert job.data["status"] == "baseline_completed"
    assert job.data["accepted"] is None


@pytest.mark.parametrize("group", ["C", "D"])
def test_experiment_logs_stay_with_experiment(group, tmp_path):
    from test_core import FakeAPI, Detection, Tool
    experiment=experimental_pipeline()
    source=tmp_path/"input.png";Image.new("RGB",(64,64)).save(source)
    job=experiment.run(source,"restore",Settings(),FakeAPI(accept=True),Detection(),Tool(),tmp_path/"experiments/runs",group)
    assert job.data["status"]=="completed"
    assert job.record_path==job.path/"record.json"
    assert job.events_path==job.path/"events.jsonl"
    assert job.record_path.is_file() and job.events_path.is_file()
    assert not (job.path/"record.log").exists()
