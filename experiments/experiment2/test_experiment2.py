import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from agentrestoration.core import Analysis, Settings, Verdict, Job, prepare, digest, write_json
from .protocol import evaluate
from .runner import run_trial
from .report import build_report, summarize


class API:
    provenance = "simulated"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def analyze(self, original, requirement, job):
        self.calls.append("analysis")
        job.event("api_request",kind="vlm",model="fake")
        return Analysis(degradations=[dict(type="noise",severity="moderate",uncertain=False)],
                        restoration_goal="denoise",preserve=[],edit_prompt="denoise")

    def edit(self, original, prompt, job):
        self.calls.append("edit")
        job.reserve("image")
        job.event("api_request",kind="image",model="fake")
        if self.fail:
            raise TimeoutError()
        index = job.data["image_requests_sent"]
        output = job.path/f"image_{index}.png"
        Image.new("RGB",(32,32),(100+index*10,)*3).save(output)
        return output

    def compare(self, original, candidate, checks, job):
        self.calls.append("compare")
        job.event("api_request",kind="vlm",model="fake")
        accepted = candidate.name.startswith("tool_")
        return Verdict(decision="accept" if accepted else "uncertain",quality_improved=accepted,
                       structure_preserved=True,appearance_change="normal_repair",issues=[],
                       reason="test",retry_instruction="denoise")


class Detector:
    def detect(self, path):
        return []


class Tool:
    name = "Restormer"

    def available(self, analysis):
        return True

    def run(self, original, job):
        output = job.path/"tool_1.png"
        Image.new("RGB",(32,32),(127,)*3).save(output)
        return output


def scene(tmp_path):
    source,reference = tmp_path/"source.png",tmp_path/"reference.png"
    Image.new("RGB",(32,32),(100,)*3).save(source)
    Image.new("RGB",(32,32),(128,)*3).save(reference)
    baseline = evaluate(source,reference,tmp_path/"baseline.png")
    return dict(scene="one",input=str(source),reference=str(reference),input_metrics=baseline)


def test_shared_candidates_requests_fallback_and_existing_trial(tmp_path):
    sample,api = scene(tmp_path),API()
    result = run_trial(sample,1,Settings(restoration_task="gaussian_denoising"),
                       tmp_path/"trial",Detector(),threading.Lock(),api=api,tool=Tool())
    assert result["status"] == "completed"
    a,b,c = result["rows"]
    assert api.calls.count("analysis") == 1
    assert api.calls.count("edit") == 2
    assert api.calls.count("compare") == 3
    assert {r["first_candidate_sha256"] for r in result["rows"]} == {a["output_sha256"]}
    assert [r["equivalent_image_requests"] for r in result["rows"]] == [1,2,2]
    assert [r["actual_image_requests"] for r in result["rows"]] == [1,1,0]
    assert [r["tool_runs"] for r in result["rows"]] == [0,0,1]
    assert b["status"] == "needs_review" and b["metrics"]["delta_psnr"] == 0
    assert c["accepted"] and c["metrics"]["delta_psnr"] > 0
    calls = api.calls.copy()
    assert run_trial(sample,1,Settings(),tmp_path/"trial",Detector(),threading.Lock(),
                     api=api,tool=Tool()) == result
    assert api.calls == calls
    for row in (b,c):
        data = json.loads(Path(row["record_file"]).read_text(encoding="utf-8"))
        assert data["edit_requests"][0]["actually_sent_here"] is False
        assert data["edit_requests"][0]["prompt"] == "denoise"


def test_real_request_failure_retains_used_budget(tmp_path):
    api = API(fail=True)
    result = run_trial(scene(tmp_path),1,Settings(),tmp_path/"trial",Detector(),
                       threading.Lock(),api=api,tool=Tool())
    assert result["status"] == "execution_failed"
    record = next((tmp_path/"trial/A").glob("*/record.json"))
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["image_requests_sent"] == 1 and api.calls.count("edit") == 1
    assert run_trial(scene(tmp_path),1,Settings(),tmp_path/"trial",Detector(),
                     threading.Lock(),api=api,tool=Tool()) == result
    assert api.calls.count("edit") == 1


def test_metric_mapping_ratio_gate_and_exact_match(tmp_path):
    sample = scene(tmp_path)
    enlarged = tmp_path/"enlarged.png"
    Image.new("RGB",(64,64),(127,)*3).save(enlarged)
    result = evaluate(enlarged,sample["reference"],tmp_path/"mapped.png")
    assert result["eligible"] and result["mapping"].startswith("LANCZOS")
    assert result["psnr"] > 40
    Image.new("RGB",(64,32)).save(enlarged)
    result = evaluate(enlarged,sample["reference"],tmp_path/"invalid.png")
    assert not result["eligible"] and result["psnr"] is None
    result = evaluate(sample["reference"],sample["reference"],tmp_path/"exact.png")
    assert result["perfect_match"] and result["psnr"] is None and result["ssim"] == 1


def test_report_from_saved_trial_has_physical_counts_and_figures(tmp_path):
    sample = scene(tmp_path)
    trial = tmp_path/"trials/one/repeat_01"
    result = run_trial(sample,1,Settings(),trial,Detector(),threading.Lock(),api=API(),tool=Tool())
    write_json(tmp_path/"protocol.json",dict(scenes=1,repeats=1))
    write_json(tmp_path/"manifest.json",[sample])
    summary = build_report(tmp_path)
    assert summary["actual_requests"] == dict(image=2,vlm=4)
    assert summary["completed_trials"] == 1 and summary["actual_tools"] == 1
    assert summary["paired"]["C-B"]["wins"] == 1
    assert (tmp_path/"figures/quality.png").is_file()
    assert (tmp_path/"results.csv").is_file()
    report = (tmp_path/"report.md").read_text(encoding="utf-8")
    assert "实际发送：编辑 2 次" in report
    assert "返回输入图" in report


def test_scene_aggregation_does_not_treat_repetitions_as_independent_scenes():
    rows = []
    for name,values in (("one",[10,10,10]),("two",[30])):
        for repeat,value in enumerate(values):
            rows.append(dict(scene=name,repeat=repeat,group="A",accepted=None,output_is_input=False,
                             equivalent_image_requests=1,equivalent_vlm_requests=1,tool_runs=0,
                             equivalent_seconds=1,metrics=dict(psnr=value,ssim=.5,delta_psnr=1,delta_ssim=.1)))
    assert summarize(rows)["groups"]["A"]["mean"]["psnr"] == 20


def test_report_rerun_method_and_review_progress_follow_saved_data(tmp_path,monkeypatch):
    from . import report
    sample=scene(tmp_path)
    run_trial(sample,1,Settings(),tmp_path/"trials/one/repeat_01",Detector(),
              threading.Lock(),api=API(),tool=Tool())
    write_json(tmp_path/"protocol.json",dict(scenes=1,repeats=1,rerun=dict(groups=["B","C"])))
    write_json(tmp_path/"manifest.json",[sample])
    write_json(tmp_path/"supplement/human_review_summary.json",
               dict(final_outputs=dict(unique_outputs=3,unique_annotated=2)))
    monkeypatch.setattr(report,"figures",lambda *args:[])
    report.build_report(tmp_path)
    text=(tmp_path/"report.md").read_text(encoding="utf-8")
    assert "本轮复用 A 的退化分析与首次候选" in text
    assert "人工评价已完成 2/3 张独立输出" in text
    assert "尚未填入自动统计" not in text


def test_explicit_recovery_reuses_saved_first_candidate(tmp_path):
    from .shared import SharedAPI
    from .runner import REQUIREMENT
    sample,api = scene(tmp_path),API()
    trial = tmp_path/"trial"
    first = Job(trial/"A",Settings())
    original = prepare(sample["input"],first,Settings())
    shared = SharedAPI(api,trial)
    analysis = shared.analyze(original,REQUIREMENT,first)
    candidate = shared.edit(original,analysis.edit_prompt,first)
    expected = digest(candidate)
    write_json(trial/"started.json",dict(scene="one",repeat=1))
    result = run_trial(sample,1,Settings(),trial,Detector(),threading.Lock(),
                       api=api,tool=Tool(),recover=True)
    assert result["status"] == "completed"
    assert result["first_candidate_sha256"] == expected
    assert api.calls.count("edit") == 2  # saved first + one new retry
    assert api.calls.count("analysis") == 1
    assert (trial/"recovery.json").is_file()


def test_recovery_does_not_reissue_an_unknown_image_request(tmp_path):
    import pytest
    from .shared import SharedAPI
    from .runner import REQUIREMENT
    sample,api = scene(tmp_path),API()
    trial = tmp_path/"trial"
    first = Job(trial/"A",Settings())
    original = prepare(sample["input"],first,Settings())
    shared = SharedAPI(api,trial)
    analysis = shared.analyze(original,REQUIREMENT,first)
    shared.edit(original,analysis.edit_prompt,first)
    first.reserve("image")
    first.event("api_request",kind="image",model="fake")
    write_json(trial/"started.json",dict(scene="one",repeat=1))
    calls = api.calls.copy()
    with pytest.raises(RuntimeError,match="Unknown image/tool"):
        run_trial(sample,1,Settings(),trial,Detector(),threading.Lock(),api=api,tool=Tool(),recover=True)
    assert api.calls == calls
