# AgentRestoration

单张图片修复系统：分析退化 → 生成修复指令 → 编辑图片 → 检查结构与效果 → 接受、重试或工具回退。

合作者先阅读 [项目结构与协作指南](项目结构.md)，了解目录职责、核心调用关系和修改入口。

## 运行

首次安装（Windows，先安装 Python 3.12 64 位）：

```powershell
.\install.cmd
```

也可双击 `install.cmd`。脚本在项目目录创建 `.venv`，按锁文件安装依赖，下载 YOLO11n 与三种修复工具，检查配置，并在终端隐藏输入 API 密钥。两种 API 共用密钥时，第二次输入直接回车。已有的 `api.txt` 保留。准备完成后自动打开工作台。

API 文件位于项目根目录，与 README 同级；格式见 [api.example.txt](api.example.txt)。也可先将示例复制为 `api.txt`，在“识别API”和“编辑API”各自下一行填写密钥。服务地址和模型名位于 `configs/default.json`，应与密钥对应的平台一致。安装检查不发送修复请求。

常用安装选项：

```powershell
.\install.cmd -NoStart                                  # 安装完成后暂不启动
.\install.cmd -CheckOnly -NonInteractive                 # 检查现有环境与模型资产
.\install.cmd -PythonPath 'F:\Python312\python.exe'       # 指定 Python
.\install.cmd -NonInteractive                            # 自动安装，API 通过文件或环境变量配置
```

重复安装会复用现有环境和校验通过的模型资产。下载中断后可重新执行；错误会显示在终端并停止安装。`-NonInteractive` 不提示输入或启动界面，首次会创建空白 `api.txt` 供填写。安装环境、缓存与权重均位于项目所在目录。

在项目根目录打开 PowerShell：

```powershell
.\run_ui.ps1
```

脚本自动复用已经运行的工作台，重复执行会打开现有页面。使用 `.\run_ui.ps1 -NoBrowser` 可启动或检查服务并保留浏览器当前页面。

更新代码或配置后，执行 `.\run_ui.ps1 -Restart` 重新加载服务。初始化阶段即创建任务日志，配置校验失败也会保存字段、错误类型及代码位置，页面显示对应日志目录。

浏览器打开 [本地图像修复工作台](http://127.0.0.1:8501)。左侧选择图片、填写修复要求和任务类型，中间对比原图与候选，右侧实时显示流程。成功节点为绿色，未通过节点为红色，进行中节点为蓝色，等待和跳过为灰色。重试与工具回退按每次尝试展开。

处理完成后可切换候选、显示检测框、查看实际修复指令，并下载输出图片和处理记录。

命令行运行：

```powershell
.\run.ps1 '图片路径.png'
```

已知是合成高斯噪声时，启用 Restormer 去噪回退：

```powershell
.\run.ps1 'F:\cvlab\datasets\test-data\denoising_testsets\CBSD68_25\291000.png' --task gaussian_denoising
```

参数：`--requirement` 修复要求、`--config` 配置文件、`--output` 输出目录、`--task` 任务类型。默认 `auto` 使用分析、编辑和验证；专用工具按已配置且适用的任务启用。

低光增强与去雾回退：

```powershell
.\run.ps1 '低光图片.png' --task low_light
.\run.ps1 '有雾图片.png' --task dehazing
```

工作台任务选项包含“自动处理”“合成高斯去噪”“低光增强”和“去雾”。自动处理会根据明确的低光或雾退化启用对应工具；选择任务类型后，仅启用与任务及退化分析相符的工具。

也可以直接调用 Python：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m agentrestoration.cli '图片路径.png'
```

## 处理结果

每次任务的图片、分析和指令保存在 `runs/<任务ID>/`，日志集中在 `log/tasks/<同一任务ID>/`。`record.log` 保存完整任务记录，内容为 JSON；`events.log` 按行追加 JSON 事件。记录中的 `artifact_directory` 关联图片目录。每次实际编辑指令保存在图片目录的 `prompts/image_1.txt`、`prompts/image_2.txt`，并通过 `edit_requests` 关联调用序号和候选。验证通过后生成 `final.png`。

在 PowerShell 中搜索系统任务错误：

```powershell
Select-String -Path .\log\tasks\*\events.log -Pattern 'api_error|execution_failed'
```

使用 `rg` 搜索：

```powershell
rg -n 'api_error|execution_failed' .\log\tasks -g events.log
```

命令行返回图片目录和日志目录。界面下载处理记录时提供 `record.log`。实验和联调记录保存在各自原有目录。

状态：`completed` 表示接受修复结果，`needs_review` 表示保留原图和候选供复核，`execution_failed` 表示执行错误。每任务最多两次编辑请求、一次专用工具流程；重试和回退均从原图开始，每个候选都重新验证。

## 环境与配置

环境在 `.venv/`，权重在 `weights/`，缓存和临时文件在 `.cache/`，均位于 F 盘。当前使用 CPU 版 PyTorch。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir F:\Agentrestoration\.cache\pip -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/setup_assets.py
```

识图模型为 `qwen3-vl-plus`，编辑模型为 `qwen-image-2.1-turbo`。凭据从本地 `api.txt` 读取，或使用 `VLM_API_KEY`、`IMAGE_API_KEY` 环境变量；环境变量优先。地址和参数见 `configs/default.json`。运行时图片会发送至配置的外部 API。

专用工具使用作者权重：

| 工具 | 权重 | 任务 |
| --- | --- | --- |
| Restormer | 彩色高斯盲去噪 | `gaussian_denoising` |
| Retinexformer | LOL-v1 | `low_light` |
| DehazeFormer | 作者演示版 MCT，混合数据训练 | `dehazing` |

工具统一由 `restoration_tools.py` 选择，共用每任务一次工具流程的额度，结果进入结构和效果验证。混合退化优先匹配程度更重的适用任务。权重位于 `weights/`，作者架构、许可及来源记录位于 `third_party/<工具名>/`。Restormer 和 Retinexformer 使用分块推理；DehazeFormer-MCT 在 256×256 预览上预测映射参数，再对整图应用曲线。MCT 坐标张量已适配输入设备，改动与原始文件哈希记录在 `source.json`。

单独准备新增工具：

```powershell
.\.venv\Scripts\python.exe scripts/setup_assets.py --tool retinexformer
.\.venv\Scripts\python.exe scripts/setup_assets.py --tool dehazeformer
```

新增工具从作者 Hugging Face 发布地址下载权重，推理使用现有 PyTorch、einops 环境。模型来源见 [Retinexformer 作者仓库](https://github.com/caiyuanhao1998/Retinexformer) 与 [DehazeFormer 作者演示](https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo)。

编辑接口输入限制通过 `image_api_max_bytes` 配置，默认 10 MB。较大的基准图会单独生成适合发送的 JPEG，优先保留分辨率；发送尺寸、体积和编码参数保存在请求记录中，原图和结构检查基准图保持原样。API 错误码、脱敏说明和请求 ID 写入任务事件，供页面与日志查看。

界面使用 Streamlit，通过 `ui` 可选依赖管理；本机依赖版本已包含在锁文件中。单独安装界面依赖可使用：

```powershell
.\.venv\Scripts\python.exe -m pip install --cache-dir F:\Agentrestoration\.cache\pip "streamlit>=1.67,<2"
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/agentrestoration/` | 完整修复闭环与命令行入口 |
| `ui/` | 三栏工作台、流程图和图片预览 |
| `configs/` | 系统配置 |
| `scripts/` | 模型资产准备 |
| `tests/` | 系统测试 |
| `runs/` | 系统运行结果 |
| `log/tasks/` | 系统任务的过程日志与完整记录 |
| `experiments/` | 独立实验代码、文档、数据和结果 |

测试命令：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q
```

系统文档：[项目结构](项目结构.md)、[需求文档](需求文档.md)、[开发方案](开发方案.md)。实验内容统一见 [实验目录](experiments/README.md)。
