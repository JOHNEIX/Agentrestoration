"""Link imported human labels to final outputs and first-candidate gate decisions."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]

from agentrestoration.core import digest, write_json
from experiments.experiment2.review import Label, make_samples, review_summary


def gate_summary(attempts):
    usable = lambda a: a["content_error"] == "no" and a["quality"] == "improved"
    good = [a for a in attempts if usable(a)]
    errors = [a for a in attempts if a["content_error"] == "yes"]
    rejected_good = sum(a["gate_status"] == "rejected" for a in good)
    accepted_errors = sum(a["gate_status"] == "accepted" for a in errors)
    return dict(trials=len(attempts), accepted=sum(a["gate_status"]=="accepted" for a in attempts),
                rejected=sum(a["gate_status"]=="rejected" for a in attempts),
                human_usable=len(good), rejected_usable=rejected_good,
                rejected_usable_rate=rejected_good/len(good) if good else None,
                human_content_errors=len(errors), accepted_content_errors=accepted_errors,
                caught_content_errors=sum(a["gate_status"]=="rejected" for a in errors))


def analyze(directory):
    directory = Path(directory).resolve()
    trials = json.loads((directory/"results.json").read_text(encoding="utf-8"))
    rows = [r for t in trials for r in t["rows"]]
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    samples = make_samples(rows,manifest)
    source = directory/"supplement/human_labels.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["run_id"] != directory.name:
        raise ValueError("Human labels belong to a different run")
    labels = {}
    for sample in samples:
        label = payload["labels"].get(sample["sample_id"])
        if label is None:
            continue
        label = Label.model_validate(label).model_dump()
        if (label["sample_id"],label["output_sha256"]) != (sample["sample_id"],sample["output_sha256"]):
            raise ValueError("Human label does not match output")
        labels[sample["sample_id"]] = label
    by_output = {(s["scene"],s["output_sha256"]):labels.get(s["sample_id"],
                 dict(sample_id=s["sample_id"],content_error="pending",quality="pending",notes="")) for s in samples}
    final = review_summary(rows,samples,labels)
    for group in "ABC":
        part = [by_output[r["scene"],r["output_sha256"]] for r in rows if r["group"]==group]
        final["groups"][group]["quality"] = dict(Counter(l["quality"] for l in part))
    attempts = []
    for row in rows:
        if row["group"] == "A":
            continue
        record_path = Path(row["record_file"])
        record = json.loads(record_path.read_text(encoding="utf-8"))
        candidate = next(c for c in record["candidates"] if c["id"]=="image_1")
        sha = digest(record_path.parent/candidate["path"])
        if sha != row["first_candidate_sha256"]:
            raise ValueError("First candidate hash mismatch")
        label = by_output[row["scene"],sha]
        verdict = candidate.get("verdict",{})
        checks = candidate.get("checks",{})
        attempts.append(dict(scene=row["scene"],repeat=row["repeat"],group=row["group"],
                             candidate_sha256=sha,**label,gate_status=candidate["status"],
                             structure_status=checks.get("status"),structure_reason=checks.get("reason"),
                             effective_decision=verdict.get("decision"),
                             reason=verdict.get("reason",checks.get("reason","")),
                             record_file=str(record_path),record_sha256=digest(record_path)))
    summary = dict(run_id=directory.name,final_outputs=final,
                   first_gate={g:gate_summary([a for a in attempts if a["group"]==g]) for g in "BC"},
                   definitions=dict(usable="content_error=no AND quality=improved",
                                    rejected_usable_rate="rejected usable first candidates / all usable first candidates",
                                    gate="effective pipeline decision, including detector and appearance checks",
                                    counting="paired scene-repeat trials; shared B/C observations are not independent"),
                   source=dict(results_sha256=digest(directory/"results.json"),labels_sha256=digest(source),
                               code_sha256=digest(Path(__file__))),restoration_api_calls=0,vlm_api_calls=0)
    folder = directory/"supplement"
    write_json(folder/"human_review_summary.json",summary)
    write_json(folder/"reviewed_gate_attempts.json",attempts)
    with (folder/"reviewed_gate_attempts.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(attempts[0]))
        writer.writeheader()
        writer.writerows(attempts)
    lines = ["# 实验二人工复核结果","",
             f"{final['unique_outputs']} 张独立输出，已复核 {final['unique_annotated']} 张。同一张图的标签回填到关联的组别与重复。","",
             "人工标注分别记录内容观察与质量改善；系统按修复要求判断主要内容、构图和明显伪影，允许正常修复中的细微纹理、表情和颗粒差异。","",
             "## 最终输出","",
             "| 方法 | 错误标记 | 无明显错误 | 无法确认 | 待复核 | 标记率 | 改善 | 无明显变化 | 质量待评 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g,info in final["groups"].items():
        rate = f"{100*info['content_error_rate']:.2f}%" if info["content_error_rate"] is not None else "待复核"
        q = info["quality"]
        lines.append(f"| {g} | {info['yes']} | {info['no']} | {info['uncertain']} | {info['pending']} | {rate} | {q.get('improved',0)} | {q.get('unchanged',0)} | {q.get('pending',0)} |")
    lines += ["","标记率为 content_error=yes /（yes＋no），无法确认与待复核单列。内容标注与质量改善分别统计。","",
              "## 首次候选的验证行为","",
              "通过候选文件哈希连接人工标注和实际执行记录。此处统计闭环的最终验证决定，包含结构检查、外观规则及 VLM。","",
              "| 路线 | 首次接受 | 首次未接受 | 人工认为有改善且无明显错误 | 其中未接受 | 人工错误标记 | 其中接受 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for g,info in summary["first_gate"].items():
        lines.append(f"| {g} | {info['accepted']} | {info['rejected']} | {info['human_usable']} | {info['rejected_usable']} | {info['human_content_errors']} | {info['accepted_content_errors']} |")
    error_scenes = {s["scene"] for s in samples if labels.get(s["sample_id"],{}).get("content_error")=="yes"}
    error_outputs = sum(l["content_error"]=="yes" for l in labels.values())
    lines += ["","B/C 按相同场景与重复配对，共用首次候选及可复用验证，各路线分别统计。",
              f"人工错误标记涉及 {error_outputs} 张输出、{len(error_scenes)} 个原始场景，具体观察列于下表。","",
              "## 标注详情","",
              "| 编号 | 场景 | 关联结果 | content_error | 修复质量 | 人工备注 |","| --- | --- | --- | --- | --- | --- |"]
    for sample in samples:
        label = labels.get(sample["sample_id"],{})
        if label.get("content_error") not in ("yes","uncertain"):
            continue
        members = ", ".join(f"{m['group']} 第{m['repeat']}次" for m in sample["members"])
        notes = label.get("notes","").replace("|","\\|").replace("\n"," ")
        lines.append(f"| {sample['sample_id']} | {sample['scene']} | {members} | {label['content_error']} | {label['quality']} | {notes} |")
    lines += ["","## 结果说明",""]
    for g,info in summary["first_gate"].items():
        rate = f"{100*info['rejected_usable_rate']:.2f}%" if info["rejected_usable_rate"] is not None else "不适用"
        lines.append(f"- {g}：人工认为有改善且无明显内容错误的首次候选 {info['human_usable']} 次，其中未被接受 {info['rejected_usable']} 次，占 {rate}。带有人工错误标记的首次候选 {info['human_content_errors']} 次，其中被接受 {info['accepted_content_errors']} 次。")
    lines += ["","候选逐条决定和理由见 reviewed_gate_attempts.csv。人工认可口径为 content_error=no 且 quality=improved，未接受比例按这部分首次候选计算。",
              f"本次覆盖 {len({r['scene'] for r in rows})} 个场景，图像、人工选择、备注和验证结果通过输出哈希关联。"]
    (folder/"human_review_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    target = ROOT/"experiments/experiment2/results"/directory.name/"supplement"
    target.mkdir(parents=True,exist_ok=True)
    for name in ("human_review_summary.json","human_review_report.md","reviewed_gate_attempts.csv","reviewed_gate_attempts.json","human_labels.json"):
        shutil.copyfile(folder/name,target/name)
    for parent in (folder,target):
        report = parent/"report.md"
        if report.exists():
            text = report.read_text(encoding="utf-8").split("<!-- human-review-link -->")[0].rstrip()
            report.write_text(text+"\n\n<!-- human-review-link -->\n\n人工标注和验证对照详见 [人工复核报告](human_review_report.md)。\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("HUMAN_REVIEW_REPORT="+str(target/"human_review_report.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    analyze(parser.parse_args().directory)
