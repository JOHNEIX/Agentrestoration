"""Generate an honest report from completed real experiments only."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=root/"experiments/results.json"
rows=json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
lines=["# 实验记录", "", "本文件由实际落盘结果生成，未完成样本不计入。人工标签均待复核。", "",
       f"已完成 {len(rows)}/48 组任务。样本清单：8 张校准、12 张测试，固定种子 20261112。", "",
       "| 组别 | 已完成 | 编辑请求数 | 工具启动数 | 自动接受 | 总生成流程耗时（秒） |",
       "| --- | --- | --- | --- | --- | --- |"]
for group in "ABCD":
    rs=[r for r in rows if r["group"]==group]
    accepted=str(sum(r["accepted"] is True for r in rs)) if group in "CD" else "不适用"
    lines.append(f"| {group} | {len(rs)} | {sum(r['images'] for r in rs)} | {sum(r['tools'] for r in rs)} | {accepted} | {sum(r['seconds'] for r in rs):.1f} |")
lines += ["", "## 质量指标（仅尺寸对应的输出）", "",
          "| 组别 | 可计算样本数 | 平均 PSNR | 平均 SSIM | 离线 VLM 接受/拒绝/不确定 |",
          "| --- | --- | --- | --- | --- |"]
for group in "ABCD":
    rs=[r for r in rows if r["group"]==group]
    ms=[r["metrics"] for r in rs if r["metrics"].get("psnr") is not None]
    q=Counter(r["offline_verdict"]["decision"] for r in rs)
    psnr=f"{statistics.fmean(m['psnr'] for m in ms):.4f}" if ms else "不适用"
    ssim=f"{statistics.fmean(m['ssim'] for m in ms):.4f}" if ms else "不适用"
    lines.append(f"| {group} | {len(ms)} | {psnr} | {ssim} | {q['accept']}/{q['reject']}/{q['uncertain']} |")
lines += ["", "## 证据与限制", "",
    "- 完成状态、调用次数、逐次候选及路由位于每个任务的 record.json 和 events.jsonl。A/B 完成生成不等于验证通过。",
    "- 统一离线 VLM 比较在生成后单独执行，不计入上表生成流程耗时；不能作为人工质量结论。",
    "- 校准集仅 3 个可靠检测匹配，位置通过阈值按亮度控制试验取 0.02；置信度、面积和严重阈值仍为工程初值。正式可靠性结论需要更多校准样本。",
    "- 检测类别混淆可能引发无法确认。明确无法确认的候选不自动接受；需人工评估误拒绝。",
    "- 模型可能改变分辨率；尺寸不对应时 PSNR/SSIM 留空，没有拉伸或裁剪对齐。各组可计算子集不同，存在选择偏差，不能直接据其均值排名；原图保守输出也计入最终输出指标。",
    "- 人工大尺度错误率、误接受率、误拒绝率与主观质量均为待复核，不能由自动接受率替代。",
    "- 数据配对已完成文件名、尺寸、低频空间一致性与噪声残差检查，20/20 数值一致；检查详情位于 pairing_checks.json，人工内容标注仍待复核。",
    "- 四组试运行由同一实验进程加载的实现顺序执行，配置保存在各任务中。期间最终 CLI 补充了工具子进程隔离、显式高斯任务选择和高置信度计数规则，并单独验证；本表不能当作这些后续加固行为的严格完整复跑基准。",
    "- 本轮运行采用 CPU 推理及重叠分块 Restormer，不能与论文原始硬件/全图结果直接比较。", ""]
smoke=root/"experiments/tool_smoke.json"
if smoke.exists():
    d=json.loads(smoke.read_text(encoding="utf-8"))
    lines += ["## 独立工具推理", "", f"Restormer 实际推理目录：`{d['directory']}`。",
              f"该样本 PSNR={d['metrics']['psnr']:.4f}，SSIM={d['metrics']['ssim']:.4f}；仅表示独立工具测试，不代表整个系统已接受。", ""]
(root/"experiments/实验记录.md").write_text("\n".join(lines),encoding="utf-8")
with (root/"experiments/human_review.csv").open("w",newline="",encoding="utf-8-sig") as f:
    writer=csv.writer(f);writer.writerow(["scene","group","directory","quality","large_scale_error","false_accept","false_reject","notes"])
    for r in rows: writer.writerow([r["scene"],r["group"],r["directory"],"pending","pending","pending","pending",""])
print(f"Report written: {len(rows)}/48 real experiments")
