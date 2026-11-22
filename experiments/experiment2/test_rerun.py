import json
from pathlib import Path
import shutil
import threading

import pytest

from agentrestoration.core import Settings,Verdict,digest,write_json
from . import runner
from .rerun import prepare
from .test_experiment2 import API,Detector,Tool,scene


def old_run(tmp_path):
    source=tmp_path/"old"
    source.mkdir()
    sample=scene(tmp_path)
    sample.update(input_sha256=digest(sample["input"]),reference_sha256=digest(sample["reference"]))
    (source/"input_metrics").mkdir()
    shutil.copyfile(sample["input_metrics"]["evaluated_image"],source/"input_metrics"/Path(sample["input"]).name)
    settings=Settings(restoration_task="gaussian_denoising")
    trial=source/"trials/one/repeat_01"
    result=runner.run_trial(sample,1,settings,trial,Detector(),threading.Lock(),api=API(),tool=Tool())
    write_json(source/"results.json",[result])
    write_json(source/"manifest.json",[sample])
    write_json(source/"protocol.json",dict(scenes=1,repeats=1,settings=settings.model_dump(),requirement=runner.REQUIREMENT))
    return source,sample,settings,result


def test_bc_rerun_imports_only_a_and_never_reuses_old_verification(tmp_path,monkeypatch):
    source,sample,settings,old=old_run(tmp_path)
    monkeypatch.setattr(runner,"source_hashes",lambda:{})
    target=prepare(source,settings,destination=tmp_path/"new")
    trial=target/"trials/one/repeat_01"
    entries=json.loads((trial/"shared_operations.json").read_text(encoding="utf-8"))
    assert sorted(e["operation"] for e in entries.values())==["analyze","edit"]
    assert all(Path(e["source_record"]).is_relative_to(target) for e in entries.values())
    Path(old["rows"][0]["output"]).unlink()
    class AcceptAPI(API):
        def compare(self,*args):
            self.calls.append("compare")
            args[-1].event("api_request",kind="vlm",model="fake")
            return Verdict(decision="accept",quality_improved=True,structure_preserved=True,
                           appearance_change="normal_repair",issues=[],reason="fresh",retry_instruction="")
    api=AcceptAPI()
    new_sample=json.loads((target/"manifest.json").read_text(encoding="utf-8"))[0]
    result=runner.run_trial(new_sample,1,settings,trial,Detector(),threading.Lock(),api=api,tool=Tool())
    assert result["status"]=="completed"
    assert api.calls==["compare"]
    assert all(r["output_sha256"]==old["rows"][0]["output_sha256"] for r in result["rows"])
    assert all(r["actual_image_requests"]==0 for r in result["rows"])
    assert result["rows"][1]["accepted"] and result["rows"][2]["accepted"]


def test_bc_rerun_rejects_modified_a_candidate(tmp_path,monkeypatch):
    source,_,settings,old=old_run(tmp_path)
    monkeypatch.setattr(runner,"source_hashes",lambda:{})
    Path(old["rows"][0]["output"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="A candidate changed"):
        prepare(source,settings,destination=tmp_path/"new")
