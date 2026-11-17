import json

import pytest
from pydantic import ValidationError

from agentrestoration.core import Settings, error_details


def test_validation_details_exclude_inputs():
    try:
        Settings(timeout="private-input-value")
    except ValidationError as error:
        details = error_details(error)
    assert details["type"] == "ValidationError"
    assert details["validation"][0]["field"] == "timeout"
    assert details["traceback"][0]["function"] == "test_validation_details_exclude_inputs"
    assert "private-input-value" not in json.dumps(details)


@pytest.mark.parametrize("invalid", [{"unknown_setting": "private-input-value"}, {"timeout": -1}])
def test_initialization_failure_has_task_logs(tmp_path, monkeypatch, invalid):
    from ui import service
    monkeypatch.setattr(service, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.json").write_text(json.dumps(invalid), encoding="utf-8")
    def forbidden(*args):
        pytest.fail("No API should be constructed for invalid configuration")
    monkeypatch.setattr(service, "Qwen", forbidden)
    updates = []
    job = service.execute("unused.png", "restore", "auto", lambda event, data: updates.append(data))
    assert job.data["status"] == "execution_failed"
    assert job.data["image_requests_sent"] == job.data["tool_pipeline_started"] == 0
    assert job.record_path.parent == tmp_path / "log/tasks" / job.path.name
    assert job.record_path.is_file() and job.events_path.is_file()
    record = json.loads(job.record_path.read_text(encoding="utf-8"))
    assert record["error_details"]["validation"][0]["field"] == next(iter(invalid))
    assert record["steps"][0]["id"] == "initialization" and record["steps"][0]["status"] == "failed"
    assert any(e["stage"] == "execution_failed" for e in record["events"])
    assert updates[-1]["record_file"] == str(job.record_path.resolve())
    assert "private-input-value" not in job.record_path.read_text(encoding="utf-8")


def test_initialized_job_is_reused_for_full_pipeline(tmp_path, monkeypatch):
    from PIL import Image
    from test_core import FakeAPI, Detection
    from ui import service
    monkeypatch.setattr(service, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "Qwen", lambda settings: FakeAPI(accept=True))
    monkeypatch.setattr(service, "Detector", lambda settings: Detection())
    source = tmp_path / "input.png"
    Image.new("RGB", (64,64)).save(source)
    job = service.execute(source, "restore", "auto", None)
    assert job.data["status"] == "completed"
    assert len(list((tmp_path / "runs").iterdir())) == 1
    assert [step["id"] for step in job.data["steps"]][:2] == ["initialization", "prepare"]
    assert job.record_path.parent.parent == tmp_path / "log/tasks"
