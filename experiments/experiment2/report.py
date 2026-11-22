"""Scene-level paired statistics and standalone figures from saved results."""
import csv
import json
from pathlib import Path

import numpy as np

from agentrestoration.core import write_json


def interval(values, seed=20261112):
    values = np.asarray(values,dtype=float)
    if not len(values):
        return None
    samples = np.random.default_rng(seed).choice(values,(2000,len(values)),replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples,[.025,.975])]


def summarize(rows):
    summary = dict(groups={},paired={})
    for group in "ABC":
        part = [r for r in rows if r["group"] == group]
        valid = [r for r in part if r["metrics"]["psnr"] is not None]
        scenes = sorted({r["scene"] for r in valid})
        means = {key:[float(np.mean([r["metrics"][key] for r in valid if r["scene"]==scene]))
                      for scene in scenes] for key in ("psnr","ssim","delta_psnr","delta_ssim")}
        summary["groups"][group] = dict(
            trials=len(part),metric_trials=len(valid),metric_scenes=len(scenes),
            perfect_trials=sum(r["metrics"].get("perfect_match",False) for r in part),
            mean={key:float(np.mean(values)) if values else None for key,values in means.items()},
            psnr_ci=interval(means["psnr"]),ssim_ci=interval(means["ssim"]),
            degraded_trials=sum(r["metrics"]["delta_psnr"] < -1e-9 for r in valid),
            accepted=sum(r["accepted"] is True for r in part) if group!="A" else None,
            original_returns=sum(r["output_is_input"] for r in part),
            equivalent_images=sum(r["equivalent_image_requests"] for r in part),
            equivalent_vlm=sum(r["equivalent_vlm_requests"] for r in part),
            tools=sum(r["tool_runs"] for r in part),
            mean_equivalent_seconds=float(np.mean([r["equivalent_seconds"] for r in part])) if part else None)
    for first,second in (("A","B"),("B","C")):
        lookup = {(r["scene"],r["repeat"],r["group"]):r for r in rows}
        by_scene = {}
        for scene,repeat,group in lookup:
            if group != first or (scene,repeat,second) not in lookup:
                continue
            a,b = lookup[scene,repeat,first],lookup[scene,repeat,second]
            if a["metrics"]["psnr"] is None or b["metrics"]["psnr"] is None:
                continue
            by_scene.setdefault(scene,[]).append(
                {k:b["metrics"][k]-a["metrics"][k] for k in ("psnr","ssim")})
        per_scene = {s:{k:float(np.mean([v[k] for v in values])) for k in ("psnr","ssim")}
                     for s,values in by_scene.items()}
        deltas = [v["psnr"] for v in per_scene.values()]
        ssim = [v["ssim"] for v in per_scene.values()]
        summary["paired"][second+"-"+first] = dict(
            scenes=len(per_scene),trials=sum(len(v) for v in by_scene.values()),
            mean_delta_psnr=float(np.mean(deltas)) if deltas else None,
            mean_delta_ssim=float(np.mean(ssim)) if ssim else None,
            psnr_ci=interval(deltas),ssim_ci=interval(ssim),
            wins=sum(v>1e-9 for v in deltas),ties=sum(abs(v)<=1e-9 for v in deltas),
            losses=sum(v < -1e-9 for v in deltas),per_scene=per_scene)
    return summary


def figures(directory, rows, summary, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .protocol import pixels
    folder = directory/"figures"
    folder.mkdir(exist_ok=True)
    colors = ["#89a6cd","#4391d3","#36a477"]
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,key,title in zip(axes,("psnr","ssim"),("PSNR (dB)","SSIM")):
        for i,group in enumerate("ABC"):
            info = summary["groups"][group]
            value,ci = info["mean"][key],info[key+"_ci"]
            if value is None:
                continue
            ax.bar(i,value,color=colors[i],width=.55)
            if ci:
                ax.errorbar(i,value,yerr=[[max(0,value-ci[0])],[max(0,ci[1]-value)]],
                            fmt="none",color="#334155",capsize=4)
            ax.text(i,value,f"{value:.3f}",ha="center",va="bottom",fontsize=10)
        ax.set_xticks(range(3),list("ABC"))
        ax.set_title(title+"; scene bootstrap 95% CI")
        ax.grid(axis="y",alpha=.18)
    fig.savefig(folder/"quality.png",dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(2,1,figsize=(12,6),layout="constrained")
    for ax,(comparison,info),color in zip(axes,summary["paired"].items(),colors[1:]):
        scenes = sorted(info["per_scene"])
        ax.bar(range(len(scenes)),[info["per_scene"][s]["psnr"] for s in scenes],color=color)
        ax.axhline(0,color="#475569",linewidth=.8)
        ax.set_xticks(range(len(scenes)),scenes,rotation=45,ha="right",fontsize=8)
        ax.set_ylabel("Delta PSNR (dB)")
        ax.set_title(comparison+"; average over repetitions for each scene")
    fig.savefig(folder/"paired_deltas.png",dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    groups = summary["groups"]
    axes[0].bar(list("ABC"),[groups[g]["equivalent_images"]/max(1,groups[g]["trials"]) for g in "ABC"],color=colors)
    axes[0].set_title("Equivalent editing requests per route")
    axes[1].bar(list("ABC"),[groups[g]["mean_equivalent_seconds"] or 0 for g in "ABC"],color=colors)
    axes[1].set_title("Estimated standalone route seconds")
    for ax in axes: ax.grid(axis="y",alpha=.18)
    fig.savefig(folder/"resources.png",dpi=180)
    plt.close(fig)
    lookup = {(r["scene"],r["repeat"],r["group"]):r for r in rows}
    pairs = []
    for (scene,repeat,group),b in lookup.items():
        if group != "B" or (scene,repeat,"C") not in lookup: continue
        c = lookup[scene,repeat,"C"]
        if b["metrics"]["psnr"] is not None and c["metrics"]["psnr"] is not None:
            pairs.append((c["metrics"]["psnr"]-b["metrics"]["psnr"],scene,repeat))
    selected = sorted(pairs,reverse=True)[:2]+sorted(pairs)[:1]
    gallery = []
    for _,scene,repeat in dict.fromkeys(selected):
        if not all((scene,repeat,g) in lookup for g in "ABC"): continue
        sample = next(m for m in manifest if m["scene"] == scene)
        fig,axes = plt.subplots(1,5,figsize=(16,4),layout="constrained")
        panels = [("Noisy input",sample["input"]),("Reference",sample["reference"])]
        for group in "ABC":
            row = lookup[scene,repeat,group]
            value = row["metrics"]["psnr"]
            panels.append((f"{group}; PSNR "+(f"{value:.2f}" if value is not None else "NA"),
                           row["metrics"].get("evaluated_image",row["output"])))
        for ax,(title,path) in zip(axes,panels):
            ax.imshow(pixels(path))
            ax.set_title(title,fontsize=10)
            ax.axis("off")
        name = f"case_{scene}_repeat_{repeat}.png"
        fig.savefig(folder/name,dpi=150)
        plt.close(fig)
        gallery.append(name)
    return gallery


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "—"


def build_report(directory):
    directory = Path(directory)
    protocol = json.loads((directory/"protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    trials = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((directory/"trials").glob("*/*/result.json"))]
    rows = [r for trial in trials for r in trial["rows"]]
    summary = summarize(rows)
    # Count physical sends from append-only events, including interrupted jobs.
    events = []
    for path in (directory/"trials").glob("*/*/[ABC]/*/events.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    actual = {kind:sum(e["stage"]=="api_request" and e.get("kind")==kind
                      for e in events) for kind in ("image","vlm")}
    summary.update(completed_trials=sum(t["status"]=="completed" for t in trials),
                   failed_trials=sum(t["status"]!="completed" for t in trials),
                   planned_trials=protocol["scenes"]*protocol["repeats"],actual_requests=actual,
                   actual_tools=sum(e["stage"]=="budget_reserved" and e.get("source")=="tool" for e in events),
                   recovered_trials=len(list((directory/"trials").glob("*/*/recovery.json"))))
    write_json(directory/"results.json",trials)
    write_json(directory/"summary.json",summary)
    with (directory/"results.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        fields = ["scene","repeat","group","status","psnr","ssim","delta_psnr","delta_ssim",
                  "equivalent_image_requests","actual_image_requests","tool_runs",
                  "equivalent_seconds","record_file","output"]
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = dict(row,**row["metrics"])
            writer.writerow({k:data.get(k) for k in fields})
    gallery = figures(directory,rows,summary,manifest)
    baseline_note = ("本轮复用 A 的退化分析与首次候选，B/C 按本轮源码快照重新执行验证、重试和工具回退。"
                     "A 的生成配置见 baseline_protocol.json，本轮配置见 protocol.json。") if protocol.get("rerun") else (
                     "每个试次生成一份退化分析和首次候选，A/B/C 共享该次输入与首次结果。")
    human_note = "人工内容评价由离线复核页面记录，并按输出哈希关联到各组。"
    if (directory/"supplement/human_review_summary.json").is_file():
        human = json.loads((directory/"supplement/human_review_summary.json").read_text(encoding="utf-8"))["final_outputs"]
        human_note = (f"人工评价已完成 {human['unique_annotated']}/{human['unique_outputs']} 张独立输出，"
                      "详见 [人工复核报告](supplement/human_review_report.md)。")
    lines = ["# 实验二：单次编辑、验证重试与工具回退","",
             f"数据：CBSD68_25，{protocol['scenes']} 个场景，每场景 {protocol['repeats']} 次。",
             f"已完成 {summary['completed_trials']}/{summary['planned_trials']} 个三组配对试次；执行失败 {summary['failed_trials']} 个。","",
             "A：分析后单次编辑；B：结构与效果验证、最多两次编辑；C：B 的规则加适用工具回退。三组共用首次候选，相同输入、指令与检查的模型操作在试次内复用。","",
             baseline_note,"",
             "## 最终输出质量","",
             "| 组别 | 已完成输出 / 指标有效 | 场景数 | 平均 PSNR | 平均 SSIM | ΔPSNR | ΔSSIM | PSNR 下降试次 | 返回输入图 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for group,info in summary["groups"].items():
        m = info["mean"]
        lines.append(f"| {group} | {info['trials']} / {info['metric_trials']} | {info['metric_scenes']} | "
                     f"{fmt(m['psnr'])} | {fmt(m['ssim'])} | {fmt(m['delta_psnr'])} | {fmt(m['delta_ssim'])} | "
                     f"{info['degraded_trials']}/{info['metric_trials']} | {info['original_returns']} |")
    lines += ["","先对同场景重复取平均，再对场景平均。Δ 指标相对于各自退化输入。原图回退按实际输出参与评价。","",
              "## 逐场景配对差值","",
              "| 比较 | 配对场景 / 试次 | 平均 ΔPSNR | PSNR 差值 95% 区间 | 平均 ΔSSIM | 胜 / 平 / 负场景 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for comparison,info in summary["paired"].items():
        ci = info["psnr_ci"]
        interval_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
        lines.append(f"| {comparison} | {info['scenes']} / {info['trials']} | {fmt(info['mean_delta_psnr'])} | "
                     f"{interval_text} | {fmt(info['mean_delta_ssim'])} | {info['wins']} / {info['ties']} / {info['losses']} |")
    lines += ["","置信区间按场景进行 2000 次配对 bootstrap，种子 20261112。","",
              "## 调用与时间","",
              "| 组别 | 独立执行等效编辑请求 | 等效 VLM 请求 | 工具启动 | 平均等效流程秒数 |",
              "| --- | --- | --- | --- | --- |"]
    for group,info in summary["groups"].items():
        lines.append(f"| {group} | {info['equivalent_images']} | {info['equivalent_vlm']} | {info['tools']} | {fmt(info['mean_equivalent_seconds'],2)} |")
    lines += ["",f"实际发送：编辑 {actual['image']} 次，VLM {actual['vlm']} 次；实际工具启动 {summary['actual_tools']} 次。",
              "等效计数包含复用调用；实际请求只按 api_request 事件统计。等效耗时以实测流程时间补回复用操作原耗时，并扣除工具排队时间，是独立执行时间的估计。指标计算与绘图不计入修复流程。","",
              "## 评价规则","",
              "- 完整输出与干净参考图比较，RGB、data_range=255，SSIM 使用 7×7 窗、彩色通道和样本协方差。",
              "- 宽高比相对误差小于 1% 后，统一用 LANCZOS 映射到参考尺寸；原始输出和映射记录同时保存。宽高比失败的指标留空并计入无效输出。",
              "- source_snapshot、protocol.json、manifest.json 固定源码、配置、样本与哈希；每试次 shared_operations.json 记录复用来源。",
              "- " + human_note,"",
              "## 图表","",
              "![最终质量](figures/quality.png)","",
              "![逐场景差值](figures/paired_deltas.png)","",
              "![调用与时间](figures/resources.png)","",
              "## 案例","",
              "按 C−B 的 PSNR 差值选取最高的两个试次和最低的一个试次，重复案例去重。"]
    for name in gallery:
        lines += ["",f"![{name}](figures/{name})"]
    if (directory/"supplement/report.md").is_file():
        lines += ["","<!-- supplement-link -->","",
                  "补充评价见 [LPIPS、挽救率与内容复核](supplement/report.md)。"]
    (directory/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("REPORT="+str(directory/"report.md"),flush=True)
    return summary
