import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agentrestoration.adapters import Qwen
from agentrestoration.core import Settings, Job, Analysis, Verdict
from agentrestoration.pipeline import run


class Detector:
    def detect(self, path): return []


class Tool:
    def available(self, analysis): return False


def png():
    stream=BytesIO(); Image.new("RGB", (64,64), "white").save(stream, format="PNG")
    return stream.getvalue()


class API(Qwen):
    def __init__(self, settings, timeout=False):
        super().__init__(settings)
        self.keys={"image":"test-key", "vlm":"test-key"}
        self.prompts=[]
        def respond(request):
            if request.method == "POST":
                self.prompts.append(json.loads(request.content)["prompt"])
                if timeout: raise httpx.ReadTimeout("test", request=request)
                return httpx.Response(200, json={"id":"test-request", "data":[{"url":"https://example.test/image.png"}]})
            return httpx.Response(200, content=png())
        self.client=httpx.Client(transport=httpx.MockTransport(respond))
        self.comparisons=0
    def analyze(self, *args):
        return Analysis(degradations=[],restoration_goal="denoise",preserve=[],edit_prompt="remove noise")
    def compare(self, *args):
        self.comparisons+=1
        accepted=self.comparisons>1
        return Verdict(decision="accept" if accepted else "reject",quality_improved=accepted,
                       structure_preserved=True,appearance_change="normal_repair",
                       issues=[],reason="test",retry_instruction="preserve texture")


def test_actual_prompts_and_progress(tmp_path):
    image=tmp_path/"in.png";image.write_bytes(png())
    api=API(Settings());events=[]
    job=run(image,"denoise",Settings(),api,Detector(),Tool(),tmp_path/"jobs",
            on_event=lambda event, snapshot:events.append((event,snapshot)), log_root=tmp_path/"log/tasks")
    assert job.data["status"]=="completed"
    assert len(api.prompts)==2
    assert api.prompts[0]!=api.prompts[1]
    assert "preserve texture" in api.prompts[1]
    for i,prompt in enumerate(api.prompts,1):
        record=job.data["edit_requests"][i-1]
        assert record["prompt"]==prompt==(job.path/f"prompts/image_{i}.txt").read_text(encoding="utf-8")
        assert record["candidate"]==f"image_{i}.png"
    stages={s["id"]:s for s in job.data["steps"]}
    assert stages["image_1:quality"]["status"]=="failed"
    assert stages["image_2:quality"]["status"]=="success"
    assert stages["result"]["status"]=="success"
    running=[(e,s) for e,s in events if e.get("node_id")=="image_1:structure" and e.get("status")=="running"]
    assert running and "checks" not in running[0][1]["candidates"][0]
    assert any(e["stage"]=="candidate_ready" for e,_ in events)


def test_timeout_retains_sent_prompt(tmp_path):
    p=tmp_path/"in.png";p.write_bytes(png());job=Job(tmp_path/"jobs",Settings())
    with pytest.raises(Exception): API(Settings(),timeout=True).edit(p,"exact instruction",job)
    assert job.data["image_requests_sent"]==1
    assert job.data["edit_requests"][0]["status"]=="failed"
    assert (job.path/"prompts/image_1.txt").read_text(encoding="utf-8")=="exact instruction"


def test_observer_failure_does_not_retry_or_mutate(tmp_path):
    from test_core import FakeAPI, Tool as FakeTool
    p=tmp_path/"in.png";p.write_bytes(png())
    def broken(event,snapshot):
        snapshot["image_requests_sent"]=100
        raise ValueError("presentation error")
    job=run(p,"test",Settings(),FakeAPI(accept=True),Detector(),FakeTool(),tmp_path/"jobs",on_event=broken,
            log_root=tmp_path/"log/tasks")
    assert job.data["status"]=="completed"
    assert job.data["image_requests_sent"]==1
    assert job.data["observer_errors"]


def test_task_logs_are_separate_from_images(tmp_path):
    from test_core import FakeAPI, Tool as FakeTool
    image=tmp_path/"in.png";image.write_bytes(png())
    updates=[]
    job=run(image,"restore",Settings(),FakeAPI(accept=True),Detector(),FakeTool(),tmp_path/"images",
            on_event=lambda event,snapshot:updates.append(snapshot),log_root=tmp_path/"log/tasks")
    assert job.data["status"]=="completed"
    assert job.log_path==tmp_path/"log/tasks"/job.path.name
    assert {p.name for p in job.log_path.iterdir()}=={"record.log","events.log"}
    assert not (job.path/"record.json").exists() and not (job.path/"events.jsonl").exists()
    record=json.loads(job.record_path.read_text(encoding="utf-8"))
    persisted_events=[json.loads(line) for line in job.events_path.read_text(encoding="utf-8").splitlines()]
    assert persisted_events==record["events"]
    assert record["image_requests_sent"]==1
    assert (Path(record["artifact_directory"])/record["output"]).is_file()
    assert Path(record["record_file"])==job.record_path.resolve()
    assert Path(record["events_file"])==job.events_path.resolve()
    assert updates[-1]["record_file"]==record["record_file"]
