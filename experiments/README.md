# 实验

系统代码位于 `src/agentrestoration/`，执行完整修复闭环。本目录独立组织对比方案、校准、评测、报告和实验产物。

| 文件 | 用途 |
| --- | --- |
| `runner.py` | 样本清单、校准、A/B/C/D 实验和指标计算 |
| `pipeline.py` | 实验专用流程，保留分组与基线分支 |
| `verify_pairs.py` | 数据配对检查 |
| `report.py` | 生成本目录的实验记录和待复核表 |
| `tool_smoke.py` | 专用工具的独立推理验证 |
| `runs/` | 实验候选、原图与决策记录 |
| `results.json` | 48 组历史结果及任务目录索引 |

阅读顺序：[实验设计说明](实验设计说明.md) → [实验操作流程](实验操作流程.md) → [实验记录](实验记录.md)。

命令从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe experiments/runner.py manifest
.\.venv\Scripts\python.exe experiments/verify_pairs.py
.\.venv\Scripts\python.exe experiments/runner.py calibrate
.\.venv\Scripts\python.exe experiments/runner.py run
```

通过 `results.json` 和实验记录查看已有结果。开始新一轮实验前，按操作流程归档上一轮资料；`run` 发送真实请求并写入本轮汇总。`report.py` 生成报告并重建人工复核表，已填写标签另存归档。
