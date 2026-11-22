import json
from pathlib import Path

from PIL import Image
import pytest

from .review import make_samples, import_labels, review_summary, write_review
from .supplement import compute_lpips, perceptual_summary, rescue_and_risk


def fixtures(tmp_path):
    source = tmp_path/"input.png"
    reference = tmp_path/"reference.png"
    Image.new("RGB",(32,32),(90,)*3).save(source)
    Image.new("RGB",(32,32),(128,)*3).save(reference)
    output = tmp_path/"output.png"
    Image.new("RGB",(32,32),(127,)*3).save(output)
    manifest = [dict(scene="one",input=str(source),reference=str(reference))]
    rows = []
    for group in "ABC":
        rows.append(dict(scene="one",repeat=1,group=group,output=str(output),output_sha256="shared",
                         accepted=group!="B",metrics=dict(evaluated_image=str(output),eligible=True,
                         psnr=20 if group=="B" else 30,ssim=.5 if group=="B" else .9,
                         delta_psnr=0 if group=="B" else 10)))
    return rows,manifest


def test_duplicate_outputs_review_once_but_count_in_every_group(tmp_path):
    rows,manifest = fixtures(tmp_path)
    samples = make_samples(rows,manifest)
    assert len(samples)==1 and len(samples[0]["members"])==3
    pending = review_summary(rows,samples,{})
    assert all(g["content_error_rate"] is None for g in pending["groups"].values())
    assert all(g["pending"]==1 for g in pending["groups"].values())
    label = dict(sample_id=samples[0]["sample_id"],output_sha256="shared",content_error="yes",
                 quality="improved",notes="test")
    path = tmp_path/"labels.json"
    path.write_text(json.dumps(dict(run_id="run",labels=[label])),encoding="utf-8")
    labels = import_labels(path,"run",samples)
    summary = review_summary(rows,samples,labels)
    assert summary["unique_annotated"]==1
    assert all(g["content_error_rate"]==1 for g in summary["groups"].values())


@pytest.mark.parametrize("change", ["run","hash","unknown","duplicate"])
def test_label_import_rejects_wrong_provenance(tmp_path,change):
    rows,manifest = fixtures(tmp_path)
    samples = make_samples(rows,manifest)
    label = dict(sample_id=samples[0]["sample_id"],output_sha256="shared",content_error="no")
    payload = dict(run_id="run",labels=[label])
    if change=="run": payload["run_id"]="other"
    if change=="hash": label["output_sha256"]="wrong"
    if change=="unknown": label["sample_id"]="missing"
    if change=="duplicate": payload["labels"].append(dict(label))
    path = tmp_path/"labels.json"
    path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError):
        import_labels(path,"run",samples)


def test_lpips_dedup_cache_and_scene_average(tmp_path):
    rows,manifest = fixtures(tmp_path)
    class Scorer:
        metadata = dict(metric="simulated")
        calls = []
        def __call__(self,a,b):
            self.calls.append(str(a))
            return .5 if Path(a).name=="input.png" else .1
    scorer = Scorer()
    scored,_ = compute_lpips(rows,manifest,tmp_path,scorer)
    assert len(scorer.calls)==2  # one input and one shared output
    assert all(r["lpips_gain"]==.4 for r in scored)
    compute_lpips(rows,manifest,tmp_path,scorer)
    assert len(scorer.calls)==2
    summary = perceptual_summary(scored)
    assert summary["groups"]["A"]["mean_lpips"]==.1
    assert summary["paired"]["C-B"]["ties"]==1


def test_rescue_requires_accepted_output_and_both_metrics(tmp_path):
    rows,_ = fixtures(tmp_path)
    info = rescue_and_risk([dict(rows=rows)])
    assert info["rescued"]==1 and info["B_not_accepted"]==1
    rows[2]["accepted"]=False
    assert rescue_and_risk([dict(rows=rows)])["rescued"]==0
    rows[2]["accepted"]=True
    rows[2]["metrics"]["ssim"]=.4
    assert rescue_and_risk([dict(rows=rows)])["rescued"]==0


def test_review_package_hides_groups_and_preserves_template_labels(tmp_path):
    import zipfile
    rows,manifest = fixtures(tmp_path)
    samples = make_samples(rows,manifest)
    folder = tmp_path/"review"
    write_review(folder,"run",samples,{})
    html = (folder/"review.html").read_text(encoding="utf-8")
    payload = json.loads(html.split("const data=",1)[1].split(";\nconst key=",1)[0])
    assert "members" not in payload["samples"][0] and "group" not in payload["samples"][0]
    template = folder/"human_review_template.csv"
    template.write_text("user supplied labels",encoding="utf-8")
    write_review(folder,"run",samples,{})
    assert template.read_text(encoding="utf-8")=="user supplied labels"
    with zipfile.ZipFile(tmp_path/"content_review.zip") as archive:
        assert "mapping.csv" not in archive.namelist()
        assert "review.html" in archive.namelist()


def test_cli_import_reuses_scores_without_loading_model(tmp_path,monkeypatch):
    import sys
    from agentrestoration.core import digest
    from . import supplement
    rows,manifest = fixtures(tmp_path)
    directory = tmp_path/"run"
    directory.mkdir()
    folder = directory/"supplement"
    folder.mkdir()
    (directory/"results.json").write_text(json.dumps([dict(status="completed",rows=rows)]),encoding="utf-8")
    (directory/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    scored = [dict(r,lpips=.1,input_lpips=.5,lpips_gain=.4) for r in rows]
    fingerprints = dict(results_sha256=digest(directory/"results.json"),manifest_sha256=digest(directory/"manifest.json"))
    (folder/"scored_rows.json").write_text(json.dumps(dict(rows=scored,metric_method=dict(simulated=True),fingerprints=fingerprints)),encoding="utf-8")
    sample = make_samples(rows,manifest)[0]
    labels = tmp_path/"labels.json"
    labels.write_text(json.dumps(dict(run_id="run",labels=[dict(sample_id=sample["sample_id"],output_sha256="shared",content_error="no")])),encoding="utf-8")
    monkeypatch.setattr(supplement,"ROOT",tmp_path)
    def forbidden_model():
        raise AssertionError("Label import must reuse existing scores")
    monkeypatch.setattr(supplement,"OfficialLPIPS",forbidden_model)
    monkeypatch.setattr(sys,"argv",["supplement",str(directory),"--labels",str(labels)])
    supplement.main()
    summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    assert summary["human_review"]["unique_annotated"]==1
    assert all(g["content_error_rate"]==0 for g in summary["human_review"]["groups"].values())
    assert (tmp_path/"experiments/experiment2/results/run/supplement/report.md").is_file()


def test_human_gate_summary_separates_content_errors_and_improvement():
    from experiments.analyze_experiment2_review import gate_summary
    attempts = [dict(content_error="no",quality="improved",gate_status="rejected"),
                dict(content_error="no",quality="improved",gate_status="accepted"),
                dict(content_error="yes",quality="improved",gate_status="accepted"),
                dict(content_error="uncertain",quality="unchanged",gate_status="rejected"),
                dict(content_error="no",quality="unchanged",gate_status="rejected")]
    result = gate_summary(attempts)
    assert result["human_usable"]==2
    assert result["rejected_usable"]==1 and result["rejected_usable_rate"]==.5
    assert result["human_content_errors"]==1 and result["accepted_content_errors"]==1
    assert result["caught_content_errors"]==0
