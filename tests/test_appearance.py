import json

import numpy as np
import pytest
from PIL import Image

from agentrestoration.core import ROOT, Job, Settings, Verdict, appearance_evidence, resolve_verdict


def accepted(change="normal_repair"):
    return Verdict(decision="accept",quality_improved=True,structure_preserved=True,
                   appearance_change=change,issues=[],reason="mock assessment",retry_instruction="")


def test_new_color_block_has_location_and_blocks_unresolved_accept(tmp_path):
    source=tmp_path/"source.png";candidate=tmp_path/"candidate.png"
    Image.new("RGB",(200,200),(130,150,170)).save(source)
    image=Image.open(source);image.paste((180,40,210),(100,20,180,120));image.save(candidate)
    evidence=appearance_evidence(source,candidate,(80,0,200,160))
    assert evidence["status"] == "suspect"
    box=evidence["regions"][0]["box"]
    assert .45 < box[0] < .55 and .05 < box[1] < .2
    assert resolve_verdict(accepted(),evidence).decision=="uncertain"
    assert not resolve_verdict(accepted(),evidence).accepted()
    # An explicitly requested recoloring can pass; the sentinel is evidence.
    assert resolve_verdict(accepted("user_requested"),evidence).accepted()
    assert resolve_verdict(accepted("unexpected"),evidence).decision=="reject"


@pytest.mark.parametrize("mode", ["denoise","brighten","unchanged_purple","small_patch"])
def test_normal_changes_do_not_raise_large_new_color_alert(tmp_path,mode):
    clean=np.full((160,160,3),(60,100,140),dtype=np.float32)
    if mode=="denoise":
        a=np.clip(clean+np.random.default_rng(42).normal(0,25,clean.shape),0,255).astype("uint8")
        b=clean.astype("uint8")
    elif mode=="brighten":
        a=(clean*.25).astype("uint8");b=clean.astype("uint8")
    elif mode=="unchanged_purple":
        a=np.full((160,160,3),(180,40,210),dtype="uint8");b=a.copy()
    else:
        a=clean.astype("uint8");b=a.copy();b[30:35,30:35]=(180,40,210)
    source=tmp_path/"a.png";candidate=tmp_path/"b.png"
    Image.fromarray(a).save(source);Image.fromarray(b).save(candidate)
    assert appearance_evidence(source,candidate)["status"]=="clear"


def test_missing_or_unknown_appearance_cannot_accept():
    old=accepted().model_dump();old.pop("appearance_change")
    legacy=Verdict.model_validate(old)
    assert not legacy.accepted()
    assert resolve_verdict(legacy,{"status":"clear"}).decision=="uncertain"
    old["appearance_change"]="anything"
    with pytest.raises(ValueError): Verdict.model_validate(old)
    verdict=accepted()
    resolve_verdict(verdict,{"status":"suspect"})
    assert verdict.decision=="accept"  # original VLM result stays available for logging


def test_prompt_contains_original_request_and_program_region(tmp_path,monkeypatch):
    from agentrestoration.adapters import Qwen
    source=tmp_path/"a.png";candidate=tmp_path/"b.png"
    Image.new("RGB",(160,160)).save(source);Image.new("RGB",(160,160),"white").save(candidate)
    job=Job(tmp_path/"runs",Settings())
    job.data["requirement"]="Brighten and shift the scene to warmer colors."
    job.data["analysis"]={"restoration_goal":"brighten"}
    checks={"local_repair":{"box":[40,40,120,120]},"appearance":{"status":"clear","regions":[]}}
    def vlm(paths,instruction,schema,passed_job):
        assert job.data["requirement"] in instruction
        assert "foreground AND background" in instruction
        assert "Pixel differences alone are not a failure" in instruction
        assert "new-color alert is evidence, not an automatic rejection" in instruction
        assert len(paths)==4 and paths[0]!=source
        assert np.array_equal(np.asarray(Image.open(source)),np.zeros((160,160,3),dtype="uint8"))
        assert (job.path/"validation"/"b_overview_0.png").is_file()
        return accepted("user_requested")
    api=Qwen(Settings());monkeypatch.setattr(api,"vlm",vlm)
    assert api.compare(source,candidate,checks,job).accepted()


def test_conflicting_vlm_runs_normal_budget_and_preserves_original(tmp_path):
    from test_core import FakeAPI,Detection
    from agentrestoration.pipeline import run
    class API(FakeAPI):
        def edit(self,original,prompt,job):
            job.reserve("image")
            path=job.path/f"image_{job.data['image_requests_sent']}.png"
            with Image.open(original) as image:
                image.paste((180,40,210),(30,30,130,130));image.save(path)
            return path
        def compare(self,original,candidate,checks,job):
            assert checks["appearance"]["status"]=="suspect"
            return accepted()
    class NoTool:
        def available(self,*args):return False
    source=tmp_path/"input.png";Image.new("RGB",(160,160),(130,150,170)).save(source)
    job=run(source,"denoise",Settings(),API(),Detection(),NoTool(),tmp_path/"runs",log_root=tmp_path/"log")
    assert job.data["status"]=="needs_review"
    assert job.data["image_requests_sent"]==2
    assert job.data["tool_pipeline_started"]==0
    assert not (job.path/"final.png").exists()
    for candidate in job.data["candidates"]:
        assert candidate["vlm_verdict"]["decision"]=="accept"
        assert candidate["verdict"]["decision"]=="uncertain"
    record=json.loads(job.record_path.read_text(encoding="utf-8"))
    assert any(e["stage"]=="verdict_resolved" for e in record["events"])


def test_recorded_purple_failure_is_detected_if_available():
    root=ROOT/"runs/20261122_152737_25f90def95"
    if not root.exists():pytest.skip("Local real-API regression artifact is not distributed")
    evidence=appearance_evidence(root/"original_normalized.png",root/"final.png",(256,41,455,294))
    assert evidence["status"]=="suspect"
    assert not resolve_verdict(accepted(),evidence).accepted()


@pytest.mark.parametrize("change,expected_tool", [("unexpected",1),("user_requested",0)])
def test_appearance_route_uses_existing_tool_budget(tmp_path,change,expected_tool):
    from test_core import FakeAPI,Detection
    from agentrestoration.pipeline import run
    class API(FakeAPI):
        def edit(self,original,prompt,job):
            job.reserve("image")
            output=job.path/"image_1.png"
            with Image.open(original) as image:
                image.paste((180,40,210),(30,30,130,130));image.save(output)
            return output
        def compare(self,original,candidate,checks,job):
            return accepted("normal_repair" if candidate.name=="tool_1.png" else change)
    class Tool:
        name="Test tool"
        def available(self,*args):return True
        def run(self,original,job):
            output=job.path/"tool_1.png"
            with Image.open(original) as image:image.save(output)
            return output
    source=tmp_path/"input.png";Image.new("RGB",(160,160),(130,150,170)).save(source)
    requirement="Replace the background with a flat color." if change=="user_requested" else "Remove noise."
    job=run(source,requirement,Settings(),API(),Detection(),Tool(),tmp_path/"runs",log_root=tmp_path/"log")
    assert job.data["status"]=="completed"
    assert job.data["image_requests_sent"]==1
    assert job.data["tool_pipeline_started"]==expected_tool
