"""Image workspace with settings on the left and a process panel below."""
from pathlib import Path
import uuid

import streamlit as st

from agentrestoration.core import ROOT, Settings
from agentrestoration.region import RepairRegion
from ui import region_selector, service
from ui.presentation import candidate_label, flow_html, stage_badge_html, preview_image, prompt_text, REASONS
from ui.rounds import available_choices, chosen_option, input_path


DEFAULT_REQUIREMENT = " "


def main():
    state = st.session_state
    defaults = dict(running=False, snapshot={}, upload_token=None, exiting=False,
                    repair_region=None, region_revision=0, rounds=[], selections={},
                    session_id=uuid.uuid4().hex, requirement_text=DEFAULT_REQUIREMENT)
    for key, value in defaults.items():
        state.setdefault(key, value)
    if "next_view" in state:
        state.view_round = state.pop("next_view")
    if state.pop("clear_requirement", False):
        state.requirement_text = ""

    def inherit_region(snapshot):
        state.repair_region = snapshot.get("local_repair", {}).get("selection")
        state.local_mode = state.repair_region is not None
        state.region_revision += 1

    def switch_round():
        state.snapshot = state.rounds[state.view_round]
        inherit_region(state.snapshot)

    def select_candidate(candidate_id):
        state.selections[state.snapshot["directory"]] = candidate_id
        # Coordinates describe the full image, so they also apply to full-frame candidates.
        inherit_region(state.snapshot)

    def reset_region_mode():
        state.repair_region = None
        state.region_revision += 1

    heading, actions = st.columns([5, 1], vertical_alignment="bottom")
    with heading:
        st.html('<div class="eyebrow">AGENTRESTORATION / WORKSPACE</div>')
        st.title("图像修复工作台")
    with actions:
        exit_pressed = st.button("退出工作台", key="shutdown_workbench", type="primary",
                                 use_container_width=True, disabled=state.running or state.exiting,
                                 help="停止后台服务；处理图片时请等待任务完成。")
    if exit_pressed:
        try:
            service.shutdown()
            state.exiting = True
        except RuntimeError as error:
            st.error(str(error))
    if state.exiting:
        st.success("工作台正在退出，可以关闭此页面。再次使用请运行 run_ui.ps1。")
        st.stop()
    st.html('<div class="intro">选择图片，描述修复目标，查看每一步处理，也可以选择结果继续修改。</div>')
    settings_column, workspace = st.columns([1, 3.55], gap="large")
    left = settings_column.container(border=True, key="settings_panel")
    middle = workspace.container(border=True, key="image_panel")
    right = workspace.container(border=True, key="flow_panel")

    with left:
        st.subheader("01  修复设置")
        upload = st.file_uploader("选择需要修复的图片", type=["jpg", "jpeg", "png"], disabled=state.running)
        token = upload.file_id if upload else None
        if token != state.upload_token and not state.running:
            state.update(upload_token=token, snapshot={}, rounds=[], selections={},
                         session_id=uuid.uuid4().hex, repair_region=None,
                         requirement_text=DEFAULT_REQUIREMENT)
            state.region_revision += 1
            state.pop("view_round", None)
        if state.rounds:
            st.selectbox("查看轮次", range(len(state.rounds)), key="view_round",
                         format_func=lambda i: f"第 {i+1} 轮",
                         disabled=state.running, on_change=switch_round)
        snapshot = state.snapshot
        selected = chosen_option(snapshot, state.selections.get(snapshot.get("directory")))
        continuation = selected is not None and bool(state.rounds) and bool(snapshot.get("directory"))
        requirement = st.text_area("本轮修复要求" if continuation else "修复要求",
                                   key="requirement_text", height=150, disabled=state.running)
        tasks = {"自动处理": "auto", "合成高斯去噪": "gaussian_denoising",
                 "低光增强": "low_light", "去雾": "dehazing"}
        task_label = st.selectbox("处理任务", list(tasks), disabled=state.running)
        show_boxes = st.toggle("显示检测框", value=False, disabled=state.running)
        local_mode = st.toggle("局部修复模式", key="local_mode", disabled=state.running,
                               on_change=reset_region_mode)
        if local_mode:
            st.caption("框住所需要修改的部分")
        if continuation:
            number = snapshot.get("round", {}).get("number", state.get("view_round", 0)+1)
            st.caption(f"当前基于：第 {number} 轮 · {candidate_label(selected)}")
            if not requirement.strip():
                st.caption("填写新的修复要求后继续。")
        pressed = st.button("基于所选图片继续修改" if continuation else "开始修复",
                            key="start_repair", type="primary", use_container_width=True,
                            disabled=state.running or (not continuation and upload is None)
                            or (local_mode and state.repair_region is None)
                            or (continuation and not requirement.strip()))
        st.caption("图片通过已配置的 API 进行分析与编辑。")
        counts = st.empty()

    with middle:
        st.subheader("02  图片对比")
        image_slot, detail_slot, selection_slot, download_slot = [st.empty() for _ in range(4)]
    with right:
        flow_heading, flow_status = st.columns([1, 2], vertical_alignment="center")
        with flow_heading:
            st.subheader("03  处理流程")
        with flow_status:
            current_slot = st.empty()
        graph_slot = st.empty()

    def region_picker(image, image_id):
        value = region_selector.select_region(preview_image(image), image_id, state.repair_region)
        if value and value.get("image_id") == image_id:
            region = value.get("region")
            if region is not None:
                region = RepairRegion.model_validate(region).model_dump()
            if region != state.repair_region:
                state.repair_region = region
                st.rerun()

    def render(value):
        with counts.container():
            a, b = st.columns(2)
            a.metric("编辑调用", f"{value.get('image_requests_sent', 0)} / 2")
            b.metric("工具流程", f"{value.get('tool_pipeline_started', 0)} / 1")
        current_slot.html(stage_badge_html(value))
        graph_slot.html(flow_html(value))
        candidate = chosen_option(value, None if state.running else
                                  state.selections.get(value.get("directory")))
        original = input_path(value) or upload
        checks = candidate.get("checks", {}) if candidate else {}
        with image_slot.container():
            original_col, candidate_col = st.columns(2)
            with original_col:
                st.caption("本轮输入" if state.rounds or state.running else "原图")
                if original is not None:
                    try:
                        if local_mode and not candidate and not state.running:
                            region_picker(original, f"{token}:{state.region_revision}")
                        else:
                            objects = list(checks.get("original", []) if show_boxes else [])
                            selection = value.get("local_repair", {}).get("selection")
                            if selection:
                                objects.append(dict(box=[selection[k] for k in ("x0","y0","x1","y1")],
                                                    **{"class": "修复区域"}))
                            st.image(preview_image(original, objects), width="stretch")
                    except (ValueError, OSError):
                        st.error("图片无法读取，请选择有效图片。")
                else:
                    st.html('<div class="image-empty"><span>＋</span>选择图片后在这里预览</div>')
            with candidate_col:
                st.caption(candidate.get("label", "修复候选") if candidate else "修复候选")
                if candidate:
                    if local_mode and not state.running:
                        region_picker(candidate["absolute_path"],
                                      f"{value.get('directory')}:{candidate['id']}:{state.region_revision}")
                    else:
                        st.image(preview_image(candidate["absolute_path"],
                                               checks.get("candidate", []) if show_boxes else []), width="stretch")
                else:
                    st.html('<div class="image-empty"><span>✦</span>开始处理后展示修复候选</div>')
        with detail_slot.container():
            if candidate:
                st.markdown(f"**{candidate_label(candidate)}**")
                reason = (candidate.get("verdict", {}).get("reason") or candidate.get("reason")
                          or candidate.get("error", ""))
                with st.expander("结果详情与实际修复指令", expanded=False):
                    if reason:
                        st.caption(REASONS.get(reason, reason))
                    if candidate["id"] != "original":
                        st.text(prompt_text(value["directory"], candidate, value))
                    if value.get("record_file"):
                        st.caption("任务日志：" + str(Path(value["record_file"]).parent))
            if value.get("error"):
                st.error("处理执行失败：" + value["error"])
            if value.get("record_file") and not candidate:
                st.caption("任务日志：" + str(Path(value["record_file"]).parent))
            if len(state.rounds) > 1:
                initial = input_path(state.rounds[0])
                if initial:
                    with st.expander("查看最初原图"):
                        st.image(preview_image(initial), width="stretch")

    render(snapshot)
    if not state.running:
        choices = available_choices(snapshot)
        others = [c for c in choices if selected and c["id"] != selected["id"]]
        if others:
            with selection_slot.container():
                with st.expander(f"其他候选（{len(others)}）", expanded=False):
                    columns = st.columns(min(3, len(others)))
                    for index, candidate in enumerate(others):
                        with columns[index % len(columns)]:
                            st.image(preview_image(candidate["absolute_path"]), width="stretch")
                            st.caption(candidate_label(candidate))
                            reason = candidate.get("verdict", {}).get("reason") or candidate.get("reason", "")
                            if reason:
                                st.caption(REASONS.get(reason, reason))
                            st.button("选择此图", key=f"choose:{snapshot['directory']}:{candidate['id']}",
                                      on_click=select_candidate, args=(candidate["id"],))
        if snapshot.get("status") in ("completed", "needs_review", "execution_failed"):
            with download_slot.container():
                a, b = st.columns(2)
                if selected:
                    output = selected["absolute_path"]
                    a.download_button("下载原图" if selected["id"] == "original" else "下载当前图片",
                                      output.read_bytes(), file_name=output.name,
                                      mime="image/jpeg" if output.suffix.lower() in (".jpg", ".jpeg") else "image/png",
                                      use_container_width=True)
                if snapshot.get("record_file") and Path(snapshot["record_file"]).is_file():
                    b.download_button("下载处理记录", Path(snapshot["record_file"]).read_bytes(),
                                      file_name="record.log", mime="application/json", use_container_width=True)

    if pressed:
        try:
            if continuation:
                path = selected["absolute_path"]
                origin = dict(session_id=state.session_id, number=len(state.rounds)+1,
                              parent_task_id=Path(snapshot["directory"]).name,
                              parent_candidate_id=selected["id"],
                              initial_task_id=snapshot.get("round", {}).get("initial_task_id")
                              or Path(state.rounds[0]["directory"]).name)
            else:
                settings = Settings()
                if len(upload.getvalue()) > settings.max_bytes:
                    raise ValueError("图片超过 20 MB，请选择较小的图片。")
                image = preview_image(upload)
                if image.width * image.height > settings.max_pixels:
                    raise ValueError("图片像素超过限制")
                folder = ROOT / ".cache/uploads"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / (uuid.uuid4().hex + Path(upload.name).suffix.lower())
                path.write_bytes(upload.getvalue())
                origin = dict(session_id=state.session_id, number=1)
            state.pending = dict(path=str(path), requirement=requirement, task=tasks[task_label], origin=origin)
            if local_mode:
                state.pending["region"] = RepairRegion.model_validate(state.repair_region).model_dump()
            state.running = True
            state.snapshot = dict(source_file=str(path))
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error) or "请选择有效的 JPG 或 PNG 图片。")

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
            state.snapshot.setdefault("steps", []).append(
                dict(id="result", title="最终结果", status="failed", reason=type(error).__name__))
            state.snapshot["current_stage"] = "最终结果"
        finally:
            state.running = False
        if state.snapshot.get("directory"):
            state.rounds.append(state.snapshot)
            state.next_view = len(state.rounds)-1
        state.clear_requirement = True
        st.rerun()
