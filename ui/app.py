from pathlib import Path
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

import streamlit as st
from agentrestoration.core import ROOT, Settings
from ui.presentation import candidate_label, flow_html, preview_image, prompt_text, REASONS
from ui import service

st.set_page_config(page_title="AgentRestoration · 图像修复", page_icon="✦", layout="wide")
st.html("""<style>
.stApp{background:#f5f7fb;color:#25334a}
.stMainBlockContainer{padding-top:2rem;max-width:1800px}
h1{font-size:2rem!important;letter-spacing:-.6px} h3{font-size:1.05rem!important}
div[data-testid="stColumn"]>div[data-testid="stVerticalBlockBorderWrapper"]{background:white;border-radius:16px}
.eyebrow{font:600 11px system-ui;letter-spacing:2px;color:#6483ac;margin-bottom:10px}
.intro{color:#728198;font-size:14px;margin-top:-8px;margin-bottom:28px}
button[kind="primary"]{background:#2878e3;border:0;border-radius:9px}
div[data-testid="stImage"] img{border-radius:10px;object-fit:contain}
</style>""")
st.html('<div class="eyebrow">AGENTRESTORATION / WORKSPACE</div>')
st.title("图像修复工作台")
st.html('<div class="intro">选择图片，描述修复目标，查看智能体的每一步处理。</div>')

state = st.session_state
state.setdefault("running", False)
state.setdefault("snapshot", {})
state.setdefault("upload_token", None)
snapshot = state.snapshot
left, middle, right = st.columns([1.05, 2.4, 1.3], gap="medium", border=True)

with left:
    st.subheader("01  修复设置")
    upload = st.file_uploader("选择需要修复的图片", type=["jpg", "jpeg", "png"], disabled=state.running)
    requirement = st.text_area("修复要求", "改善质量并尽量保留原始内容、构图和物体位置。", height=150, disabled=state.running)
    tasks = {"自动处理": "auto", "合成高斯去噪": "gaussian_denoising", "低光增强": "low_light", "去雾": "dehazing"}
    task_label = st.selectbox("处理任务", list(tasks), disabled=state.running)
    task = tasks[task_label]
    show_boxes = st.toggle("显示检测框", value=False, disabled=state.running)
    token = upload.file_id if upload else None
    if token != state.upload_token and not state.running:
        state.upload_token = token
        state.snapshot = snapshot = {}
        state.pop("selected_candidate", None)
    pressed = st.button("开始修复", type="primary", use_container_width=True,
                        disabled=state.running or upload is None)
    st.caption("图片通过已配置的 API 进行分析与编辑。")
    counts = st.empty()

with middle:
    st.subheader("02  图片对比")
    image_slot = st.empty()
    detail_slot = st.empty()
    selection_slot = st.empty()
    download_slot = st.empty()

with right:
    st.subheader("03  处理流程")
    current_slot = st.empty()
    graph_slot = st.empty()


def render(snapshot, selected=None):
    with counts.container():
        a, b = st.columns(2)
        a.metric("编辑调用", f"{snapshot.get('image_requests_sent', 0)} / 2")
        b.metric("工具流程", f"{snapshot.get('tool_pipeline_started', 0)} / 1")
    running = next((n for n in snapshot.get("steps", []) if n["status"] == "running"), None)
    title = running["title"] if running else snapshot.get("current_stage", "等待选择图片")
    if snapshot.get("status") == "completed":
        current_slot.success("处理完成 · 结果已接受")
    elif snapshot.get("status") in ("needs_review", "execution_failed"):
        current_slot.error("处理结束 · 需人工检查" if snapshot["status"] == "needs_review" else "处理执行失败")
    else:
        current_slot.info(title)
    graph_slot.html(flow_html(snapshot))
    candidates = [c for c in snapshot.get("candidates", []) if c.get("path")]
    candidate = candidates[selected] if selected is not None and candidates else (candidates[-1] if candidates else None)
    directory = Path(snapshot["directory"]) if snapshot.get("directory") else None
    original = directory / "original_normalized.png" if directory else None
    if original is None or not original.exists():
        original = upload
    checks = candidate.get("checks", {}) if candidate else {}
    with image_slot.container():
        original_col, candidate_col = st.columns(2)
        with original_col:
            st.caption("原图")
            if original is not None:
                try:
                    st.image(preview_image(original, checks.get("original", []) if show_boxes else []), width="stretch")
                except Exception:
                    st.error("图片无法读取，请选择有效图片。")
            else:
                st.info("选择图片后在这里预览")
        with candidate_col:
            st.caption(candidate.get("label", "修复候选") if candidate else "修复候选")
            if candidate:
                st.image(preview_image(directory / candidate["path"], checks.get("candidate", []) if show_boxes else []), width="stretch")
            else:
                st.info("开始处理后展示修复候选")
    with detail_slot.container():
        if candidate:
            st.markdown(f"**{candidate_label(candidate)}**")
            reason = candidate.get("verdict", {}).get("reason") or candidate.get("reason") or candidate.get("error", "")
            if reason:
                st.caption(REASONS.get(reason, reason))
            with st.expander("实际修复指令"):
                st.text(prompt_text(directory, candidate, snapshot))
        elif snapshot.get("error"):
            st.error("处理执行失败：" + snapshot["error"])
        if snapshot.get("record_file"):
            st.caption("任务日志：" + str(Path(snapshot["record_file"]).parent))


available = [c for c in snapshot.get("candidates", []) if c.get("path")]
selected = None
if available and not state.running:
    with selection_slot.container():
        selected = st.selectbox("查看候选", range(len(available)), index=len(available)-1,
                                format_func=lambda i: candidate_label(available[i]), key="selected_candidate")
render(snapshot, selected)

if snapshot.get("directory") and snapshot.get("status") in ("completed", "needs_review", "execution_failed") and not state.running:
    with download_slot.container():
        directory = Path(snapshot["directory"])
        output = directory / snapshot.get("output", "original_normalized.png")
        a, b = st.columns(2)
        if output.exists():
            a.download_button("下载修复图片" if snapshot.get("accepted") else "下载原图", output.read_bytes(),
                              file_name=output.name, mime="image/png", use_container_width=True)
        record_path = Path(snapshot.get("record_file", ROOT / "log/tasks" / directory.name / "record.log"))
        b.download_button("下载处理记录", record_path.read_bytes(),
                          file_name="record.log", mime="application/json", use_container_width=True)

if pressed:
    settings = Settings()
    if len(upload.getvalue()) > settings.max_bytes:
        st.error("图片超过 20 MB，请选择较小的图片。")
    else:
        try:
            preview = preview_image(upload)
            if preview.width * preview.height > settings.max_pixels:
                raise ValueError("图片像素超过限制")
            folder = ROOT / ".cache/uploads"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (uuid.uuid4().hex + Path(upload.name).suffix.lower())
            path.write_bytes(upload.getvalue())
            state.pending = dict(path=str(path), requirement=requirement, task=task)
            state.running = True
            state.snapshot = {}
            state.pop("selected_candidate", None)
            st.rerun()
        except (ValueError, OSError):
            st.error("请选择有效的 JPG 或 PNG 图片。")

if state.running and state.get("pending"):
    pending = state.pop("pending")
    def on_event(event, value):
        state.snapshot = value
        render(value)
    try:
        job = service.execute(**pending, on_event=on_event)
        state.snapshot = dict(job.data, directory=str(job.path))
    except Exception as error:
        state.snapshot.update(status="execution_failed", error=type(error).__name__)
        state.snapshot.setdefault("steps", []).append(dict(id="result", title="最终结果", status="failed", reason=type(error).__name__))
        state.snapshot["current_stage"] = "最终结果"
    finally:
        state.running = False
    st.rerun()
