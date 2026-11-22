"""Deduplicated, group-hidden human review and explicit label import."""
import csv
import json
from pathlib import Path
import random
from typing import Literal
import zipfile

from pydantic import BaseModel, ConfigDict

from agentrestoration.core import write_json
from .protocol import pixels


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    output_sha256: str
    content_error: Literal["pending","yes","no","uncertain"] = "pending"
    quality: Literal["pending","improved","unchanged","worse","uncertain"] = "pending"
    notes: str = ""


def make_samples(rows, manifest):
    lookup = {s["scene"]:s for s in manifest}
    unique = {}
    for row in rows:
        key = (row["scene"],row["output_sha256"])
        if key not in unique:
            unique[key] = dict(scene=row["scene"],output_sha256=row["output_sha256"],
                               candidate=row["metrics"].get("evaluated_image") or row["output"],
                               input=lookup[row["scene"]]["input"],
                               reference=lookup[row["scene"]]["reference"],members=[])
        unique[key]["members"].append(dict(group=row["group"],repeat=row["repeat"]))
    samples = [unique[k] for k in sorted(unique)]
    random.Random(20261112).shuffle(samples)
    for i,sample in enumerate(samples,1):
        sample["sample_id"] = f"H{i:03d}"
    return samples


def import_labels(path, run_id, samples, existing=None):
    path = Path(path)
    known = {s["sample_id"]:s for s in samples}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig",newline="") as stream:
            raw = list(csv.DictReader(stream))
        if any(r.pop("run_id","") != run_id for r in raw):
            raise ValueError("Review run ID does not match")
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("run_id") != run_id:
            raise ValueError("Review run ID does not match")
        raw = payload["labels"]
    labels = dict(existing or {})
    seen = set()
    for item in raw:
        label = Label.model_validate(item)
        if label.sample_id in seen:
            raise ValueError("Duplicate review sample")
        seen.add(label.sample_id)
        if label.sample_id not in known or known[label.sample_id]["output_sha256"] != label.output_sha256:
            raise ValueError("Review sample or output hash does not match")
        previous = labels.get(label.sample_id,{})
        current = label.model_dump()
        for field in ("content_error","quality"):
            if current[field] == "pending" and previous.get(field,"pending") != "pending":
                current[field] = previous[field]
        if not current["notes"] and previous.get("notes"):
            current["notes"] = previous["notes"]
        labels[label.sample_id] = current
    return labels


def review_summary(rows, samples, labels):
    mapping = {(s["scene"],s["output_sha256"]):s["sample_id"] for s in samples}
    groups = {}
    for group in "ABC":
        counts = dict(yes=0,no=0,uncertain=0,pending=0)
        for row in rows:
            if row["group"] != group:
                continue
            state = labels.get(mapping[row["scene"],row["output_sha256"]],{}).get("content_error","pending")
            counts[state] += 1
        known = counts["yes"]+counts["no"]
        groups[group] = dict(counts,known=known,total=sum(counts.values()),
                             content_error_rate=counts["yes"]/known if known else None)
    return dict(unique_outputs=len(samples),
                unique_annotated=sum(labels.get(s["sample_id"],{}).get("content_error","pending")!="pending"
                                     for s in samples),groups=groups)


def write_review(folder, run_id, samples, labels):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    images = folder/"images"
    images.mkdir(exist_ok=True)
    display = []
    for sample in samples:
        scene,identifier = sample["scene"],sample["sample_id"]
        paths = {}
        for name,filename in (("input",f"scene_{scene}_input.png"),
                              ("reference",f"scene_{scene}_reference.png"),
                              ("candidate",f"{identifier}.png")):
            target = images/filename
            if not target.exists():
                pixels(sample[name]).save(target)
            paths[name] = "images/"+filename
        display.append(dict(sample_id=identifier,output_sha256=sample["output_sha256"],**paths))
    default_labels = [dict(sample_id=s["sample_id"],output_sha256=s["output_sha256"],
                          content_error="pending",quality="pending",notes="") for s in samples]
    template = folder/"human_review_template.csv"
    if not template.exists():
        with template.open("w",encoding="utf-8-sig",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=["run_id",*default_labels[0]])
            writer.writeheader()
            for label in default_labels:
                writer.writerow(dict(run_id=run_id,**label))
    write_json(folder/"samples.json",samples)
    with (folder/"mapping.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id","scene","output_sha256","group","repeat"])
        for sample in samples:
            for member in sample["members"]:
                writer.writerow([sample["sample_id"],sample["scene"],sample["output_sha256"],
                                 member["group"],member["repeat"]])
    payload = json.dumps(dict(run_id=run_id,samples=display,labels=labels),
                         ensure_ascii=False).replace("<","\\u003c")
    template_html = (Path(__file__).parent/"review_template.html").read_text(encoding="utf-8")
    (folder/"review.html").write_text(template_html.replace("__REVIEW_DATA__",payload),encoding="utf-8")
    # The shareable review package deliberately omits method/group mappings.
    with zipfile.ZipFile(folder.parent/"content_review.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in [folder/"review.html",template,*sorted(images.glob("*.png"))]:
            archive.write(path,path.relative_to(folder).as_posix())
