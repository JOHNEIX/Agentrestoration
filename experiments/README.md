# 实验与工具验证

本目录组织实验二及专用工具验证，复用系统适配器和核心闭环。

实验二比较单次编辑、验证重试和工具回退，使用 20 个 CBSD68_25 场景、每场景 3 次重复。运行与统计规则见 [实验二说明](experiment2/README.md)。

已完成的本次数据见 [结果解读](experiment2/results/experiment2_20261122_222733_5d9a17/结论.md) 和 [完整报告](experiment2/results/experiment2_20261122_222733_5d9a17/report.md)。

追加的 [补充评价](experiment2/results/experiment2_20261122_222733_5d9a17/supplement/report.md) 包含 LPIPS、工具补救、质量下降统计及人工内容复核表。离线复核页面和标注导入方法见 [实验二说明](experiment2/README.md#补充评价与人工复核)。

本轮保留 A 的 60 张首次候选，B/C 已按新规则重跑完成。63 张独立输出中复用 62 份人工标注，1 张新增候选待复核。人工标注与实际验证行为见 [人工复核报告](experiment2/results/experiment2_20261122_222733_5d9a17/supplement/human_review_report.md)。

[B 非原图输出子集](experiment2/results/experiment2_20261122_222733_5d9a17/b_non_original/report.md) 排除 B 返回原图的试次，重新汇总相同试次的 A/B/C 三项质量指标。

```powershell
.\.venv\Scripts\python.exe experiments/run_experiment2.py
```

## 专用工具验证

保留 Restormer、Retinexformer、DehazeFormer 的真实推理记录及单工具验证入口。

| 文件 | 内容 |
| --- | --- |
| `check_restoration_tool.py` | 运行指定工具，保存输入、输出、哈希、尺寸和推理元数据 |
| `tool_inference_records.json` | 三种工具已有推理记录的统一索引 |
| [工具接入验证](工具接入验证.md) | 已验证权重、输入和输出说明 |
| `runs/` | 本机工具验证产物和实验二运行目录 |

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe experiments/check_restoration_tool.py test_images/01_noise15_statues.png --task gaussian_denoising
.\.venv\Scripts\python.exe experiments/check_restoration_tool.py test_images/09_low_light_kitchen.png --task low_light
.\.venv\Scripts\python.exe experiments/check_restoration_tool.py test_images/05_haze_city.jpg --task dehazing
```

脚本调用本地专用工具，支持 `--config` 和 `--output`，默认使用 `configs/default.json`，产物保存到 `experiments/runs/`。任务的 `record.json` 与 `events.jsonl` 随图片保存在同一目录。

`tool_inference_completed_unvalidated` 表示工具推理完成；完整修复任务的接受状态由系统闭环的结构检查和 VLM 效果验证决定。工具运行产物按 `.gitignore` 保存在本地，记录索引和验证入口随源码保存。
