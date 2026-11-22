"""Import A evidence only; all B/C comparisons, retries and tools run afresh."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

from agentrestoration.core import ROOT, Settings, digest, write_json
from . import runner
from .review import make_samples


def operation_key(operation, inputs):
    return hashlib.sha256(json.dumps([operation,inputs],sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def import_baseline_trial(old_trial, new_trial, row, requirement):
    old_record = Path(row["record_file"])
    record = json.loads(old_record.read_text(encoding="utf-8"))
    if digest(row["output"]) != row["output_sha256"]:
        raise ValueError("A candidate changed")
    baseline = new_trial/"baseline"/old_record.parent.name
    shutil.copytree(old_record.parent,baseline)
    candidate = baseline/Path(row["output"]).name
    original = baseline/"original_normalized.png"
    prompt = record["analysis"]["edit_prompt"]
    keys = {"analyze":operation_key("analyze",[digest(original),requirement]),
            "edit":operation_key("edit",[digest(original),prompt])}
    old_entries = json.loads((old_trial/"shared_operations.json").read_text(encoding="utf-8"))
    seeded = {}
    for operation,key in keys.items():
        entry = deepcopy(old_entries[key])
        if entry["operation"] != operation or "error" in entry:
            raise ValueError("A cache has no successful analysis/first edit")
        receipt = Path(entry["source_record"])
        receipt_target = baseline/"receipts"/(receipt.parent.name+".json")
        receipt_target.parent.mkdir(exist_ok=True)
        shutil.copyfile(receipt,receipt_target)
        if operation == "edit":
            if digest(entry["value"]) != row["output_sha256"] or digest(candidate) != row["output_sha256"]:
                raise ValueError("Cached first candidate differs from A")
            entry["value"] = str(candidate.resolve())
        else:
            if entry["value"] != record["analysis"]:
                raise ValueError("Cached analysis differs from A")
        entry["source_record"] = str(receipt_target.resolve())
        entry["imported_baseline"] = True
        seeded[key] = entry
    write_json(new_trial/"shared_operations.json",seeded)
    write_json(new_trial/"baseline_import.json",dict(
        scene=row["scene"],repeat=row["repeat"],candidate_sha256=row["output_sha256"],
        source_record_sha256=digest(old_record),source_record=str(old_record),
        retained_record=str((baseline/"record.json").resolve()),
        retained_candidate=str(candidate.resolve()),
        retained_analysis=record["analysis"],first_edit_prompt=prompt,
        original_A_metrics=row["metrics"],original_A_row=row,
        reused_operations=["analyze","first_edit"],reused_BC_operations=[]))


def prepare(source, settings, workers=2, destination=None):
    source = Path(source).resolve()
    trials = json.loads((source/"results.json").read_text(encoding="utf-8"))
    old_protocol = json.loads((source/"protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    expected = old_protocol["scenes"]*old_protocol["repeats"]
    if len(trials) != expected or any(t["status"] != "completed" for t in trials):
        raise ValueError("Source must contain all completed paired trials")
    if settings.model_dump() != old_protocol["settings"]:
        raise ValueError("Rerun must keep the original experimental settings")
    directory = Path(destination) if destination else ROOT/"experiments/runs"/(
        "experiment2_"+datetime.now().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/"input_metrics",directory/"input_metrics")
    for sample in manifest:
        if digest(sample["input"]) != sample["input_sha256"] or digest(sample["reference"]) != sample["reference_sha256"]:
            raise ValueError("Dataset files changed")
        sample["input_metrics"]["evaluated_image"] = str((directory/"input_metrics"/Path(sample["input"]).name).resolve())
    write_json(directory/"manifest.json",manifest)
    write_json(directory/"baseline_protocol.json",old_protocol)
    if (source/"source_snapshot").exists():
        shutil.copytree(source/"source_snapshot",directory/"baseline_source_snapshot")
    a_keys = set()
    for trial in trials:
        a = next(r for r in trial["rows"] if r["group"]=="A")
        old_trial = source/"trials"/a["scene"]/f"repeat_{a['repeat']:02d}"
        new_trial = directory/"trials"/a["scene"]/f"repeat_{a['repeat']:02d}"
        import_baseline_trial(old_trial,new_trial,a,old_protocol["requirement"])
        a_keys.add((a["scene"],a["output_sha256"]))
    # Human observations remain applicable only to identical scene/output pairs.
    labels_file = source/"supplement/human_labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))["labels"]
        rows = [r for t in trials for r in t["rows"]]
        samples = make_samples(rows,manifest)
        transferred = [dict(scene=s["scene"],**labels[s["sample_id"]]) for s in samples
                       if (s["scene"],s["output_sha256"]) in a_keys and s["sample_id"] in labels]
        write_json(directory/"baseline_labels.json",dict(labels=transferred,source_sha256=digest(labels_file)))
    cache_file = source/"supplement/lpips_cache.json"
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        allowed = {"input:"+m["scene"] for m in manifest} | {s+":"+h for s,h in a_keys}
        cache["values"] = {k:v for k,v in cache["values"].items() if k in allowed}
        write_json(directory/"supplement/lpips_cache.json",cache)
    protocol = deepcopy(old_protocol)
    protocol.update(created_at=datetime.now(timezone.utc).isoformat(),workers=workers,
                    source_sha256=runner.source_hashes(),settings=settings.model_dump(),
                    rerun=dict(groups=["B","C"],baseline_run=source.name,
                               baseline_results_sha256=digest(source/"results.json"),
                               imported_A_trials=len(trials),initial_generation_requests=0,
                               BC_cache_policy="No old comparisons, retries or tool outputs imported"))
    write_json(directory/"protocol.json",protocol)
    for name in protocol["source_sha256"]:
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=2)
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.source/"protocol.json").read_text(encoding="utf-8"))
    directory = prepare(args.source,Settings.model_validate(protocol["settings"]),args.workers)
    print("EXPERIMENT_DIRECTORY="+str(directory),flush=True)
    if args.prepare_only:
        return
    previous = sys.argv
    try:
        sys.argv = ["run_experiment2","--resume",str(directory)]
        runner.main()
    finally:
        sys.argv = previous
