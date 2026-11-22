"""Post-hoc LPIPS, empirical rescue rates and pending human content review."""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import warnings

import numpy as np

from agentrestoration.core import ROOT, digest, write_json
from .protocol import pixels
from .report import interval
from .review import make_samples, import_labels, review_summary, write_review


class OfficialLPIPS:
    def __init__(self):
        import lpips
        import torch
        torch.set_num_threads(4)
        self.torch = torch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=UserWarning,module=r"torchvision\.models\._utils")
            self.model = lpips.LPIPS(net="alex",version="0.1",verbose=False).cpu().eval()
        backbone = Path(torch.hub.get_dir())/"checkpoints/alexnet-owt-7be5be79.pth"
        linear = Path(lpips.__file__).parent/"weights/v0.1/alex.pth"
        backbone_hash = digest(backbone)
        if not backbone_hash.startswith("7be5be79"):
            raise ValueError("AlexNet checkpoint hash does not match the official filename")
        self.metadata = dict(metric="LPIPS",network="alex",version="0.1",package_version=version("lpips"),
                             torch_version=str(torch.__version__),torchvision_version=version("torchvision"),
                             backbone="AlexNet_Weights.IMAGENET1K_V1",backbone_sha256=backbone_hash,
                             calibration_sha256=digest(linear),device="cpu",dtype="float32",
                             normalization="RGB / 127.5 - 1",spatial=False,eval_mode=True,
                             image_size="same evaluation copy as PSNR/SSIM; no extra resizing",
                             source="https://github.com/richzhang/PerceptualSimilarity")

    def __call__(self, candidate, reference):
        a,b = pixels(candidate),pixels(reference)
        if a.size != b.size:
            raise ValueError("LPIPS image sizes differ")
        def tensor(image):
            array = np.asarray(image,dtype=np.float32)/127.5-1
            return self.torch.from_numpy(array).permute(2,0,1).unsqueeze(0)
        with self.torch.inference_mode():
            score = float(self.model(tensor(a),tensor(b)).item())
        if not np.isfinite(score):
            raise ValueError("Nonfinite LPIPS score")
        return score


def compute_lpips(rows, manifest, folder, scorer):
    folder = Path(folder)
    method = scorer.metadata
    path = folder/"lpips_cache.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(method=method,values={})
    if cache["method"] != method:
        raise ValueError("LPIPS cache method differs")
    scenes = {s["scene"]:s for s in manifest}
    def score(key, candidate, reference):
        if key not in cache["values"]:
            cache["values"][key] = dict(value=scorer(candidate,reference),
                                       candidate_sha256=digest(candidate),reference_sha256=digest(reference))
            write_json(path,cache)
        else:
            entry = cache["values"][key]
            if entry["candidate_sha256"] != digest(candidate) or entry["reference_sha256"] != digest(reference):
                raise ValueError("LPIPS cached input changed")
        return cache["values"][key]["value"]
    baseline = {}
    for scene in manifest:
        baseline[scene["scene"]] = score("input:"+scene["scene"],scene["input"],scene["reference"])
    augmented = []
    for i,row in enumerate(rows,1):
        scene = scenes[row["scene"]]
        eligible = row["metrics"]["eligible"]
        candidate = row["metrics"].get("evaluated_image")
        if eligible and not candidate:
            raise ValueError("Evaluation copy is missing")
        value = score(row["scene"]+":"+row["output_sha256"],candidate,scene["reference"]) if eligible else None
        augmented.append(dict(row,lpips=value,input_lpips=baseline[row["scene"]],
                              lpips_gain=baseline[row["scene"]]-value if value is not None else None))
        if i%15 == 0:
            print(f"LPIPS {i}/{len(rows)} outputs; cached evaluations={len(cache['values'])}",flush=True)
    return augmented,method


def perceptual_summary(rows):
    result = dict(groups={},paired={})
    for group in "ABC":
        part = [r for r in rows if r["group"]==group and r["lpips"] is not None]
        scenes = sorted({r["scene"] for r in part})
        values = [float(np.mean([r["lpips"] for r in part if r["scene"]==s])) for s in scenes]
        gains = [float(np.mean([r["lpips_gain"] for r in part if r["scene"]==s])) for s in scenes]
        result["groups"][group] = dict(trials=len(part),scenes=len(scenes),
            mean_lpips=float(np.mean(values)) if values else None,
            mean_gain_vs_input=float(np.mean(gains)) if gains else None,ci=interval(values),
            worse_than_input=sum(r["lpips_gain"] < -1e-9 for r in part))
    for first,second in (("A","B"),("B","C")):
        lookup = {(r["scene"],r["repeat"],r["group"]):r for r in rows}
        deltas = {}
        for scene,repeat,group in lookup:
            if group!=first or (scene,repeat,second) not in lookup: continue
            a,b = lookup[scene,repeat,first],lookup[scene,repeat,second]
            if a["lpips"] is not None and b["lpips"] is not None:
                deltas.setdefault(scene,[]).append(b["lpips"]-a["lpips"])
        values = [float(np.mean(v)) for v in deltas.values()]
        result["paired"][second+"-"+first] = dict(
            mean_delta=float(np.mean(values)) if values else None,ci=interval(values),scenes=len(values),
            wins=sum(v < -1e-9 for v in values),ties=sum(abs(v)<=1e-9 for v in values),
            losses=sum(v > 1e-9 for v in values))
    return result


def rescue_and_risk(trials):
    total = len(trials)
    failures = rescued = better = worse = equal = 0
    for trial in trials:
        lookup = {r["group"]:r for r in trial["rows"]}
        b,c = lookup["B"],lookup["C"]
        failed = b["accepted"] is not True
        failures += failed
        bp,cp = b["metrics"],c["metrics"]
        if any(v is None for v in (bp["psnr"],cp["psnr"],bp["ssim"],cp["ssim"])):
            continue
        positive = cp["psnr"]>bp["psnr"]+1e-9 and cp["ssim"]>bp["ssim"]+1e-9
        better += positive
        worse += cp["psnr"]<bp["psnr"]-1e-9 and cp["ssim"]<bp["ssim"]-1e-9
        equal += abs(cp["psnr"]-bp["psnr"])<=1e-9 and abs(cp["ssim"]-bp["ssim"])<=1e-9
        rescued += failed and c["accepted"] is True and positive
    risk = {}
    rows = [r for t in trials for r in t["rows"]]
    for group in "ABC":
        part = [r for r in rows if r["group"]==group and r["metrics"]["delta_psnr"] is not None]
        values = sorted(r["metrics"]["delta_psnr"] for r in part)
        n = len(values)
        bottom = max(1,int(np.ceil(n*.1)))
        risk[group] = dict(trials=n,psnr_degraded=sum(v < -1e-9 for v in values),
                           psnr_degraded_rate=sum(v < -1e-9 for v in values)/n if n else None,
                           psnr_drop_over_1db=sum(v < -1 for v in values),
                           worst_delta_psnr=min(values) if n else None,
                           bottom_10pct_mean_delta_psnr=float(np.mean(values[:bottom])) if n else None)
    return dict(total_trials=total,B_not_accepted=failures,rescued=rescued,
                rescue_rate_among_B_failures=rescued/failures if failures else None,
                rescue_rate_all=rescued/total if total else None,
                both_metrics_better=better,both_metrics_worse=worse,both_metrics_equal=equal,risk=risk)


def write_outputs(folder, rows, summary):
    folder = Path(folder)
    write_json(folder/"summary.json",summary)
    with (folder/"results_with_lpips.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        fields = ["scene","repeat","group","status","psnr","ssim","lpips","input_lpips","lpips_gain","output_sha256"]
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = dict(row,**row["metrics"])
            writer.writerow({k:data.get(k) for k in fields})
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    colors = ["#89a6cd","#4391d3","#36a477"]
    groups = summary["perceptual"]["groups"]
    axes[0].bar(list("ABC"),[groups[g]["mean_lpips"] or 0 for g in "ABC"],color=colors)
    axes[0].set_title("LPIPS-Alex v0.1 (lower is better)")
    axes[0].set_ylabel("Scene-mean perceptual distance")
    axes[0].set_ylim(0,max((groups[g]["mean_lpips"] or 0) for g in "ABC")*1.2 or 1)
    for i,g in enumerate("ABC"):
        v = groups[g]["mean_lpips"]
        if v is not None: axes[0].text(i,v+.005,f"{v:.4f}",ha="center",fontsize=10)
    risk = summary["rescue"]["risk"]
    axes[1].bar(list("ABC"),[100*risk[g]["psnr_degraded_rate"] for g in "ABC"],color=colors)
    axes[1].set_title("PSNR below noisy input")
    axes[1].set_ylabel("Trials (%)")
    axes[1].set_ylim(0,max(100*risk[g]["psnr_degraded_rate"] for g in "ABC")*1.2 or 1)
    for i,g in enumerate("ABC"):
        value = 100*risk[g]["psnr_degraded_rate"]
        axes[1].text(i,value+.5,f"{value:.1f}% ({risk[g]['psnr_degraded']}/{risk[g]['trials']})",ha="center",fontsize=10)
    for ax in axes: ax.grid(axis="y",alpha=.18)
    fig.savefig(folder/"additional_metrics.png",dpi=180)
    plt.close(fig)
    def f(value):
        return f"{value:.4f}" if value is not None else "—"
    lines = ["# 实验二补充评价","",
             f"汇总本轮 {len(rows)} 份分组输出的感知质量、工具补救、质量下降情况和人工标注。","",
             "## 感知质量：LPIPS","",
             "LPIPS-Alex v0.1 使用 ImageNet 预训练 AlexNet 和作者学习权重。RGB 映射到 [-1,1]，float32、CPU、eval 模式，与 PSNR/SSIM 共用评价副本和尺寸。数值越低越接近参考图。","",
             "| 组别 | 有效输出 / 场景 | 平均 LPIPS ↓ | 相对输入的 LPIPS 改善 ↑ | LPIPS 比输入更差的次数 |",
             "| --- | --- | --- | --- | --- |"]
    for group,info in groups.items():
        lines.append(f"| {group} | {info['trials']} / {info['scenes']} | {f(info['mean_lpips'])} | "
                     f"{f(info['mean_gain_vs_input'])} | {info['worse_than_input']} |")
    lines += ["","同场景先平均三次重复，再平均场景。改善量为输入 LPIPS 减去输出 LPIPS。","",
              "| 比较 | 平均 LPIPS 差值（负值更好） | 95% 场景 bootstrap 区间 | 胜 / 平 / 负场景 |",
              "| --- | --- | --- | --- |"]
    for name,info in summary["perceptual"]["paired"].items():
        ci = info["ci"]
        bounds = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
        lines.append(f"| {name} | {f(info['mean_delta'])} | {bounds} | {info['wins']} / {info['ties']} / {info['losses']} |")
    r = summary["rescue"]
    lines += ["","## 工具补救与质量下降统计","",
              f"- B 未接受的试次：{r['B_not_accepted']}。其中 C 接受且最终 PSNR、SSIM 同时高于 B 的试次：{r['rescued']}。",
              f"- B 未接受试次的挽救率：{100*r['rescue_rate_among_B_failures']:.2f}%；占全部试次比例：{100*r['rescue_rate_all']:.2f}%。" if r['rescue_rate_among_B_failures'] is not None else "- 本次没有 B 未接受试次，挽救率不适用。",
              f"- C 相比 B 双指标更好 / 更差 / 相同：{r['both_metrics_better']} / {r['both_metrics_worse']} / {r['both_metrics_equal']}。","",
              "| 组别 | PSNR 下降次数 / 有效试次 | 下降超过 1 dB | 最差 ΔPSNR | 最差 10% 的平均 ΔPSNR |",
              "| --- | --- | --- | --- | --- |"]
    for group,info in r["risk"].items():
        lines.append(f"| {group} | {info['psnr_degraded']} / {info['trials']} | {info['psnr_drop_over_1db']} | "
                     f"{f(info['worst_delta_psnr'])} | {f(info['bottom_10pct_mean_delta_psnr'])} |")
    h = summary["human_review"]
    lines += ["","## 人工内容标注","",
              f"{len(rows)} 份输出去重为 {h['unique_outputs']} 个场景—输出组合，匿名编号并隐藏组别；当前已标注 {h['unique_annotated']} 个。",
              "同一输出的标签按关联关系回填到各组。标记率为 content_error=yes /（yes＋no），无法确认和待复核单列。人工记录包含细节观察，系统按修复要求判断主要内容、构图和明显伪影。","",
              "| 组别 | 错误标记 | 无明显错误 | 无法确认 | 待复核 | 已明确判定 | 标记率 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for group,info in h["groups"].items():
        rate = f"{100*info['content_error_rate']:.2f}%" if info["content_error_rate"] is not None else "待复核"
        lines.append(f"| {group} | {info['yes']} | {info['no']} | {info['uncertain']} | {info['pending']} | {info['known']} / {info['total']} | {rate} |")
    lines += ["","![补充指标](additional_metrics.png)","",
              "模型版本、两组权重 SHA-256、源数据指纹和补充代码哈希见 method.json。原始逐次结果见 results_with_lpips.csv。",
              "参考：[LPIPS 作者实现](https://github.com/richzhang/PerceptualSimilarity)。"]
    (folder/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--labels",type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    trials = json.loads((directory/"results.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if not trials or any(t["status"]!="completed" for t in trials):
        raise ValueError("Supplement requires completed paired trials")
    rows = [r for t in trials for r in t["rows"]]
    folder = directory/"supplement"
    folder.mkdir(exist_ok=True)
    fingerprints = dict(results_sha256=digest(directory/"results.json"),manifest_sha256=digest(directory/"manifest.json"))
    if args.labels:
        saved = json.loads((folder/"scored_rows.json").read_text(encoding="utf-8"))
        if saved["fingerprints"] != fingerprints:
            raise ValueError("Scored experiment data changed")
        scored,metric_method = saved["rows"],saved["metric_method"]
    else:
        scored,metric_method = compute_lpips(rows,manifest,folder,OfficialLPIPS())
        write_json(folder/"scored_rows.json",dict(fingerprints=fingerprints,rows=scored,metric_method=metric_method))
    samples = make_samples(rows,manifest)
    label_file = folder/"human_labels.json"
    labels = json.loads(label_file.read_text(encoding="utf-8"))["labels"] if label_file.exists() else {}
    if args.labels:
        labels = import_labels(args.labels,directory.name,samples,labels)
        write_json(label_file,dict(run_id=directory.name,labels=labels))
    summary = dict(perceptual=perceptual_summary(scored),rescue=rescue_and_risk(trials),
                   human_review=review_summary(rows,samples,labels))
    method = dict(stage="post_hoc_additional_evaluation",time=datetime.now(timezone.utc).isoformat(),
                  restoration_api_calls=0,**fingerprints,lpips=metric_method,
                  aggregation="Average repeats within scene, then average scenes; paired bootstrap by scene",
                  rescue_definition="B not accepted; C accepted and both final PSNR/SSIM exceed B",
                  risk_definition="Delta PSNR below 0; extra threshold -1 dB; mean of worst ceil(10%) trials",
                  review_seed=20261112,source_sha256={p.name:digest(p) for p in
                  (Path(__file__),Path(__file__).with_name("review.py"),Path(__file__).with_name("review_template.html"))})
    write_json(folder/"method.json",method)
    write_outputs(folder,scored,summary)
    write_review(folder/"review",directory.name,samples,labels)
    exported = ROOT/"experiments/experiment2/results"/directory.name
    target = exported/"supplement"
    target.mkdir(parents=True,exist_ok=True)
    for name in ("report.md","summary.json","method.json","results_with_lpips.csv","additional_metrics.png"):
        shutil.copyfile(folder/name,target/name)
    for name in ("human_review_template.csv","mapping.csv"):
        shutil.copyfile(folder/"review"/name,target/name)
    for parent in (directory,exported):
        report = parent/"report.md"
        if report.exists():
            text = report.read_text(encoding="utf-8")
            marker = "<!-- supplement-link -->"
            if marker in text: text = text.split(marker)[0].rstrip()+"\n"
            report.write_text(text+"\n"+marker+"\n\n补充评价见 [LPIPS、挽救率与内容复核](supplement/report.md)。\n",encoding="utf-8")
    print("SUPPLEMENT_REPORT="+str(target/"report.md"),flush=True)
    print("HUMAN_REVIEW="+str(folder/"review/review.html"),flush=True)
    print("REVIEW_ZIP="+str(folder/"content_review.zip"),flush=True)


if __name__ == "__main__":
    main()
