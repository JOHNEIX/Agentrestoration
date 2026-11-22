# 实验二：单次编辑、验证重试与工具回退

比较三种路线的最终输出质量与处理开销。源码、测试与说明在本目录，产物写入 `experiments/runs/experiment2_<时间>_<标识>/`。

本次已完成 20 场景、3 次重复：[结果解读](results/experiment2_20261122_222733_5d9a17/结论.md)、[完整报告](results/experiment2_20261122_222733_5d9a17/report.md)、[逐试次 CSV](results/experiment2_20261122_222733_5d9a17/results.csv)、[LPIPS 与挽救率补充报告](results/experiment2_20261122_222733_5d9a17/supplement/report.md)。用于分享的报告、图表和统计保存在 `results/`，原始候选及任务日志保存在运行目录。

## 实验目的与架构对应

实验比较验证、重试和专用工具回退对最终修复质量与处理开销的影响。A 提供单次编辑基线，B 加入候选验证和编辑重试，C 增加按需工具回退。三组共享首次候选，按相同场景与重复进行比较。

本轮固定已有 A 分析与首次候选，B/C 使用修订后的验证规则：YOLO 类别、数量及位置变化作为参考，VLM 根据图像判断质量改善、主要内容与构图保持以及明显伪影。普通细微纹理、表情和颗粒变化按任务容差处理。

## 本轮结果

| 方法 | PSNR / dB ↑ | SSIM ↑ | LPIPS ↓ | 最终接受 | 返回原图 |
| --- | --- | --- | --- | --- | --- |
| A | 22.2907 | 0.6458 | 0.1278 | 直接输出 | 0 |
| B | 21.9367 | 0.6198 | 0.1591 | 57/60 | 3 |
| C | 22.5368 | 0.6515 | 0.1258 | 60/60 | 0 |

C 在场景 14037 的三次重复中均通过 Restormer 完成补救，最终接受 60/60 次。按已有人工标注，55 次有改善且无明显内容错误的首次候选中，4 次未被接受，占 7.27%。本轮新增编辑请求 4 次、VLM 请求 67 次、工具启动 3 次。

## 设置

| 项目 | 配置 |
| --- | --- |
| 数据 | CBSD68_25 与 CBSD68 同名干净参考图 |
| 场景 | 文件名排序、种子 `20261112` 打乱，选择 20 个不同场景 |
| 重复 | 每场景 3 次，共 60 个三组配对试次、180 个输出 |
| A | VLM 分析后编辑一次，直接输出首次候选 |
| B | 核心闭环验证与重试，编辑最多两次，关闭工具 |
| C | 相同核心闭环，增加适用的 Restormer 回退，工具最多一次 |
| 参数 | 本轮沿用 A 生成时的配置，任务为 `gaussian_denoising`；B/C 执行源码单独冻结 |
| 并行 | 两个独立试次并行，专用工具串行推理 |

本轮每个试次导入一份已保存的 A 分析与首次候选。A/B/C 共用该次输入与首次结果；同试次内，相同输入及指令的编辑、相同候选及检查证据的验证复用本轮结果。新建完整实验时由程序生成各试次的分析与首次候选。实验分组、共享适配器和统计位于本目录。

## 评价

- 完整输出与干净参考图按 RGB 计算 PSNR、SSIM，`data_range=255`。
- SSIM 使用彩色通道、7×7 窗口、非高斯权重和样本协方差。
- 宽高比相对误差小于 1% 后，统一使用 LANCZOS 映射到参考尺寸。原始输出保留，评价副本和映射参数单独保存；比例失败的指标留空并单列。
- 每张图计算相对输入的 ΔPSNR、ΔSSIM。返回原图也参与统计，提升为零。
- 先平均同场景的重复，再对场景平均；B−A、C−B 按相同场景和重复配对，报告胜、平、负及按场景进行 2000 次 bootstrap 的区间。
- 无限 PSNR 用 `perfect_match` 标记，有限均值和完全一致数量分别保存。

实际请求按 `api_request` 事件统计，独立执行等效计数包含共享调用。等效耗时以实测时间补回共享操作原耗时，并减去复用耗时和工具排队时间，是独立执行时间的估计。检测器预热、指标计算和绘图置于路线耗时之外。

## 命令

项目根目录运行，默认即 20 场景、3 次重复：

```powershell
.\.venv\Scripts\python.exe experiments/run_experiment2.py
```

支持 `--scenes`、`--repeats`、`--seed`、`--workers`、`--data-root` 和 `--config`。仅生成方案与样本清单：

```powershell
.\.venv\Scripts\python.exe experiments/run_experiment2.py --prepare-only
```

执行准备好的实验，跳过已完成试次：

```powershell
.\.venv\Scripts\python.exe experiments/run_experiment2.py --resume experiments/runs/<实验目录>
```

显式恢复使用 `--recover-interrupted`，核对已发送请求与缓存。编辑请求均有对应缓存、工具启动次数为 0 且核心代码与冻结版本一致时，程序复用候选继续执行，并保存恢复说明及源码修订快照。

```powershell
.\.venv\Scripts\python.exe experiments/run_experiment2.py --resume experiments/runs/<实验目录> --recover-interrupted
```

重建报告和运行模拟测试：

```powershell
.\.venv\Scripts\python.exe experiments/run_experiment2.py --report-only experiments/runs/<实验目录>
.\.venv\Scripts\python.exe -m pytest experiments/experiment2/test_experiment2.py -q
```

## 保留 A，重跑 B/C

```powershell
.\.venv\Scripts\python.exe experiments/rerun_experiment2.py experiments/runs/<已有实验目录>
```

复用原实验的场景、重复、要求、配置、退化分析与首次候选，在新目录重新执行 B/C。每个试次只导入分析和首次编辑缓存，验证、后续编辑与工具结果使用当前版本重新运行。支持 `--workers` 和 `--prepare-only`，已准备的目录通过原入口 `--resume` 执行。

`baseline/` 保留 A 图片、指令与原始调用记录；`baseline_import.json` 关联文件哈希和原始 A 指标。`baseline_protocol.json` 和 `baseline_source_snapshot/` 保存 A 的生成版本，`protocol.json` 与 `source_snapshot/` 固定当前 B/C 版本。实际新请求按当前 A/B/C 任务事件统计，历史请求记录单独保存在 baseline 中，等效资源统计包含复用操作。

已有人工标注只按同场景、相同输出哈希迁移。已有 LPIPS 缓存同样按图像和权重指纹核对，修复输出变化后重新计算。

## 补充评价与人工复核

使用本轮已保存的 180 份最终输出进行离线补充评价：

```powershell
.\.venv\Scripts\python.exe experiments/supplement_experiment2.py experiments/runs/experiment2_20261122_222733_5d9a17
```

在项目根目录安装实验依赖：`.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"`。首次计算 LPIPS 自动下载官方 AlexNet 预训练权重到项目 `.cache/torch_home/`。补充源码位于 `experiments/experiment2/`，逐图指标缓存随运行目录保存，模型权重位于项目 `.cache/torch_home/`。

- **LPIPS ↓**：作者实现的 AlexNet v0.1，ImageNet 预训练骨干加学习权重，RGB 归一化为 [-1,1]。沿用 PSNR/SSIM 的评价副本和尺寸，以场景为统计单位；记录依赖版本和两组权重哈希。
- **未接受试次挽救率 ↑**：分母为 B 最终未接受的试次；分子要求 C 最终接受，且 PSNR、SSIM 均高于 B。另报占全部试次的比例。
- **质量下降统计 ↓**：相对退化输入的 PSNR 下降次数、下降超过 1 dB 次数、最差改善量及最差 10% 试次的平均改善量。
- **人工内容标注**：分别记录 `content_error`、修复质量和备注；标记率为 yes /（yes＋no），无法确认与待复核单列。相同场景、相同输出哈希标注一次，回填关联组别与重复。人工记录包含细节观察，系统验收按主要内容、构图和明显伪影判断。

人工复核步骤：

1. 打开运行目录下 `supplement/review/review.html`，或将 `supplement/content_review.zip` 发给合作者，解压后打开页面。
2. 页面并排显示退化输入、干净参考和候选；候选使用 H001 等匿名编号，不显示 A/B/C。点击图片查看原尺寸。
3. 判断内容错误和相对输入的修复质量，必要时填写区域与原因。物体增删、形状或文字改变、异常色块、背景内容改变计为内容错误；正常去噪、平滑和锐化结合参考判断。无法辨别时选择“无法确认”。
4. 页面自动保存到当前浏览器，完成后点击“导出标注 JSON”。CSV 模板也可填写，枚举值见下表。
5. 导入标注，更新报告；运行编号、图片编号和输出哈希会逐项核对，不重复计算 LPIPS。

```powershell
.\.venv\Scripts\python.exe experiments/supplement_experiment2.py experiments/runs/experiment2_20261122_222733_5d9a17 --labels <导出的JSON或填写后的CSV路径>
```

| 字段 | 可填写值 |
| --- | --- |
| `content_error` | `yes` 存在错误、`no` 无明显错误、`uncertain` 无法确认、`pending` 待复核 |
| `quality` | `improved` 改善、`unchanged` 无明显变化、`worse` 变差、`uncertain` 无法确认、`pending` 待复核 |
| `notes` | 对象、区域与判断理由 |

本次重跑输出去重为 63 张，按相同图像哈希复用 62 份人工标注，新增候选 H037（场景 216081，B/C 第 1 次重复的编辑重试结果）待复核。人工选择、备注和当前验证结论通过输出哈希关联。组别统计与首次候选的验证对照见 [人工复核报告](results/experiment2_20261122_222733_5d9a17/supplement/human_review_report.md)。

导入后，关联实际候选哈希与验证记录，生成详细对照：

```powershell
.\.venv\Scripts\python.exe experiments/analyze_experiment2_review.py experiments/runs/experiment2_20261122_222733_5d9a17
```

匿名图片顺序由种子 `20261112` 固定，组别映射单独保存在 `mapping.csv`，分享用 ZIP 不包含映射。复核标签来自人工，在线 VLM 接受状态仍作为独立字段保留。

## 排除 B 返回原图后的统计

已生成 [B 非原图输出子集报告](results/experiment2_20261122_222733_5d9a17/b_non_original/report.md)。筛选依据为 `B.output_is_input == true`：排除 3 个配对试次，保留 57 个试次、19 个场景，A/C 同步使用相同试次。

```powershell
.\.venv\Scripts\python.exe experiments/filter_experiment2.py experiments/runs/experiment2_20261122_222733_5d9a17
```

基于逐图 PSNR、SSIM、LPIPS 离线汇总。沿用场景等权口径：平均同场景保留的重复，再平均场景。报告包含 B 筛选前后、同一子集 A/B/C 及配对差值；JSON 另存逐试次等权均值。全量表描述全部试次，子集表描述 B 交付修复图时的质量。

## 文件

| 文件 | 内容 |
| --- | --- |
| `protocol.json`、`manifest.json` | 方案、参数、样本、哈希和数值配对检查 |
| `source_snapshot/` | 本轮实验执行时的源码快照 |
| `baseline_protocol.json`、`baseline_source_snapshot/` | 原 A 生成时的方案与源码 |
| `trials/<场景>/repeat_*/baseline/`、`baseline_import.json` | 原 A 图片、指令、调用记录、哈希与指标 |
| `source_revisions.json`、`recovery_source_*/` | 显式恢复时的实验代码修订与快照 |
| `trials/<场景>/repeat_*/A、B、C/` | 各组原图、候选、实际指令和日志 |
| 每试次 `shared_operations.json` | 共享调用、耗时和来源记录 |
| 每试次 `evaluation/` | 用于指标计算的统一尺寸副本 |
| `results.json`、`results.csv` | 逐试次与分组结果 |
| `summary.json`、`report.md` | 场景级汇总、配对统计和报告 |
| `figures/` | 质量、配对差值、资源和案例对照图 |
| `supplement/report.md`、`summary.json` | LPIPS、工具补救、质量下降统计和人工标注 |
| `supplement/method.json` | 补充评价口径、依赖与权重版本、源码及数据哈希 |
| `supplement/results_with_lpips.csv` | 逐试次 PSNR、SSIM、LPIPS 和相对输入改善量 |
| `supplement/review/`、`content_review.zip` | 匿名复核页面、图片、CSV 模板与分享包 |
| `supplement/human_labels.json` | 显式导入后的人工标签 |
| `supplement/human_review_report.md`、`human_review_summary.json` | 最终输出的人工评价及首次候选的实际验证行为 |
| `supplement/reviewed_gate_attempts.csv`、`reviewed_gate_attempts.json` | 人工标签、候选哈希、实际验证结论和理由的逐次对照 |
| `b_non_original/report.md`、`summary.json`、`results.csv` | B 非原图输出子集、排除清单、源数据哈希及同试次 A/B/C 指标 |

展示案例按 C−B 的 PSNR 差值选最高两个试次与最低一个试次，重复案例去重。全部结果保存在统计表中，人工内容判断与自动指标分别记录。
