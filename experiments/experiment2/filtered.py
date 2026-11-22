"""Paired subset evaluation selected by B's original-return flag."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from agentrestoration.core import ROOT, digest, write_json
from .report import interval

METRICS = ("psnr", "ssim", "lpips")


def value(row, metric):
    return row.get("lpips") if metric == "lpips" else row["metrics"][metric]


def select_trials(rows):
    trials = {}
    for row in rows:
        key = (row["scene"], row["repeat"])
        groups = trials.setdefault(key, {})
        if row["group"] in groups:
            raise ValueError("Duplicate group in a paired trial")
        groups[row["group"]] = row
    kept, excluded = [], []
    for (scene, repeat), groups in sorted(trials.items()):
        if set(groups) != set("ABC"):
            raise ValueError("Each trial must include A, B and C")
        returned = groups["B"].get("output_is_input")
        if not isinstance(returned, bool):
            raise ValueError("B original-return flag must be a boolean")
        if returned:
            excluded.append(dict(scene=scene, repeat=repeat,
                                 reason="B.output_is_input == true",
                                 B_output_sha256=groups["B"]["output_sha256"]))
        else:
            kept.extend(groups[g] for g in "ABC")
    return kept, excluded


def summarize_metrics(rows):
    groups = {}
    for group in "ABC":
        part = [r for r in rows if r["group"] == group]
        metrics = {}
        for metric in METRICS:
            by_scene = {}
            for row in part:
                v = value(row, metric)
                if v is not None:
                    if not np.isfinite(v):
                        raise ValueError("Nonfinite saved metric")
                    by_scene.setdefault(row["scene"], []).append(v)
            scene_means = [float(np.mean(v)) for v in by_scene.values()]
            all_values = [v for values in by_scene.values() for v in values]
            metrics[metric] = dict(mean=float(np.mean(scene_means)) if scene_means else None,
                                   trial_mean=float(np.mean(all_values)) if all_values else None,
                                   valid_trials=len(all_values), scenes=len(by_scene),
                                   ci=interval(scene_means))
        groups[group] = dict(trials=len(part), scenes=len({r["scene"] for r in part}), metrics=metrics)
    return groups


def paired_metrics(rows):
    lookup = {(r["scene"], r["repeat"], r["group"]): r for r in rows}
    comparisons = {}
    for first, second in (("A", "B"), ("B", "C")):
        metrics = {}
        for metric in METRICS:
            by_scene = {}
            for (scene, repeat, group), row in lookup.items():
                if group != first:
                    continue
                other = lookup[scene, repeat, second]
                a, b = value(row, metric), value(other, metric)
                if a is not None and b is not None:
                    by_scene.setdefault(scene, []).append(b-a)
            means = [float(np.mean(v)) for v in by_scene.values()]
            metrics[metric] = dict(mean_delta=float(np.mean(means)) if means else None,
                                   ci=interval(means), scenes=len(means),
                                   trials=sum(len(v) for v in by_scene.values()))
        comparisons[second+"-"+first] = metrics
    return comparisons


def write_report(folder, summary, retained):
    def f(v):
        return f"{v:.4f}" if v is not None else "—"
    def table_row(label, stats):
        m = stats["metrics"]
        return f"| {label} | {stats['trials']} | {stats['scenes']} | " + " | ".join(f(m[k]["mean"]) for k in METRICS) + " |"
    kept, removed = summary["retained_trial_count"], summary["excluded_trial_count"]
    lines = ["# 实验二：排除 B 返回原图试次后的统计", "",
             f"从 {summary['total_trial_count']} 个配对试次中排除 B 返回原图的 {removed} 次，保留 {kept} 次。",
             "按原始记录中的 `B.output_is_input` 筛选，A、C 同步保留相同的场景与重复，形成配对比较。", "",
             "## B 筛选前后", "",
             "| 范围 | 试次 | 场景 | PSNR ↑ / dB | SSIM ↑ | LPIPS ↓ |",
             "| --- | --- | --- | --- | --- | --- |",
             table_row("完整 B", summary["full"]["B"]),
             table_row("B 非原图输出", summary["filtered"]["B"]), "",
             "## 同一批保留试次的 A/B/C", "",
             "| 方法 | 试次 | 场景 | PSNR ↑ / dB | SSIM ↑ | LPIPS ↓ |",
             "| --- | --- | --- | --- | --- | --- |"]
    lines += [table_row(g, summary["filtered"][g]) for g in "ABC"]
    lines += ["", "沿用原实验的场景等权口径：先平均同场景中保留的重复，再平均场景。无保留重复的场景不参与。",
              "JSON 同时保存逐试次等权均值 `trial_mean`、各指标有效数和置信区间。", "",
              "| 配对比较 | 指标 | 平均差值 | 95% 场景 bootstrap 区间 |",
              "| --- | --- | --- | --- |"]
    for comparison, metrics in summary["paired"].items():
        for metric, info in metrics.items():
            ci = info["ci"]
            bounds = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
            lines.append(f"| {comparison} | {metric.upper()} | {f(info['mean_delta'])} | {bounds} |")
    lines += ["", "差值为后组减前组；PSNR、SSIM 正值更好，LPIPS 负值更好。", "",
              "本表汇总 B 非原图输出子集的质量，[全量结果](../report.md) 汇总全部试次。",
              "使用逐图指标重新汇总，图像映射与 LPIPS 权重沿用本轮评价方案。",
              "筛选清单与来源哈希保存在 summary.json，保留的逐图指标见 results.csv。"]
    (folder/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (folder/"results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["scene", "repeat", "group", "output_is_input", "psnr", "ssim", "lpips", "output_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in retained:
            writer.writerow(dict(scene=row["scene"], repeat=row["repeat"], group=row["group"],
                                 output_is_input=row["output_is_input"], output_sha256=row["output_sha256"],
                                 **{k:value(row, k) for k in METRICS}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    trials = json.loads((directory/"results.json").read_text(encoding="utf-8"))
    if not trials or any(t["status"] != "completed" for t in trials):
        raise ValueError("Filtering requires completed paired trials")
    scored = json.loads((directory/"supplement/scored_rows.json").read_text(encoding="utf-8"))
    fingerprints = dict(results_sha256=digest(directory/"results.json"), manifest_sha256=digest(directory/"manifest.json"))
    if scored["fingerprints"] != fingerprints:
        raise ValueError("Cached scores do not match source results")
    rows = [row for t in trials for row in t["rows"]]
    cached = scored["rows"]
    if len(rows) != len(cached) or any(any(c.get(k) != v for k,v in r.items()) for r,c in zip(rows,cached)):
        raise ValueError("Cached rows do not match source results")
    retained, excluded = select_trials(cached)
    summary = dict(selection="Exclude entire paired trial when B.output_is_input is true",
                   aggregation="Mean retained repeats within scene, then equal-weight mean across scenes",
                   total_trial_count=len(trials), retained_trial_count=len(retained)//3,
                   excluded_trial_count=len(excluded), excluded=excluded,
                   retained=[dict(scene=r["scene"], repeat=r["repeat"]) for r in retained if r["group"]=="B"],
                   full=summarize_metrics(cached), filtered=summarize_metrics(retained), paired=paired_metrics(retained),
                   source=dict(**fingerprints, scored_rows_sha256=digest(directory/"supplement/scored_rows.json"),
                               lpips=scored["metric_method"], code_sha256=digest(Path(__file__))),
                   restoration_api_calls=0, vlm_api_calls=0)
    folder = directory/"b_non_original"
    folder.mkdir(exist_ok=True)
    write_json(folder/"summary.json", summary)
    write_report(folder, summary, retained)
    target = ROOT/"experiments/experiment2/results"/directory.name/"b_non_original"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("report.md", "summary.json", "results.csv"):
        shutil.copyfile(folder/name, target/name)
    print((target/"report.md").read_text(encoding="utf-8"))
    print("FILTERED_REPORT="+str(target/"report.md"))


if __name__ == "__main__":
    main()
