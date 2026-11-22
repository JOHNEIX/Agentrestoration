"""Pure presentation helpers, shared by the UI and its tests."""
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

LABELS = {"accepted": "已接受", "rejected": "未通过", "executing": "正在处理", "execution_failed": "执行失败",
          "original": "已保留"}
REASONS = {"aspect_ratio": "宽高比变化", "object_position": "检测框位置差异（参考）", "matched": "主要目标匹配",
           "suspected_count_change": "目标对应关系差异（参考）", "high_confidence_suspected_count_change": "疑似目标增删",
           "unreliable_or_truncated_detection": "目标检测信息不足（参考）"}


def candidate_label(candidate):
    name = candidate.get("label") or ("工具回退" if candidate["source"] == "tool" else candidate.get("path", "编辑候选"))
    status = "需复核" if candidate.get("verdict", {}).get("decision") == "uncertain" else LABELS.get(candidate['status'], candidate['status'])
    return f"{name} · {status}"


def stage_badge_html(snapshot):
    running = next((node for node in snapshot.get("steps", []) if node["status"] == "running"), None)
    status = snapshot.get("status")
    if status == "completed":
        color, text = "success", "处理完成 · 结果已接受"
    elif status in ("needs_review", "execution_failed"):
        color = "failed"
        text = "处理结束 · 需人工检查" if status == "needs_review" else "处理执行失败"
    else:
        color = "running" if running or status in ("preparing", "processing") else "pending"
        text = running["title"] if running else snapshot.get("current_stage", "等待选择图片")
    return (f'<div class="flow-status-wrap"><div class="flow-status {color}" role="status" aria-live="polite">'
            f'<span class="status-dot" aria-hidden="true"></span><span class="status-text">{escape(text)}</span>'
            '</div></div>')


def flow_html(snapshot):
    steps = snapshot.get("steps") or [
        dict(id="prepare", title="图片准备", status="pending"),
        dict(id="analysis", title="退化分析", status="pending"),
        dict(id="image_1:generate", title="首次编辑", status="pending", attempt=1),
        dict(id="image_1:structure", title="结构检查", status="pending", attempt=1),
        dict(id="image_1:quality", title="效果验证", status="pending", attempt=1),
        dict(id="result", title="最终结果", status="pending")]
    # Preparation leads into the first attempt; the result closes the last one.
    groups, preparation = [], []
    for node in steps:
        attempt = node.get("attempt")
        if attempt is not None:
            if not groups or groups[-1][0] != attempt:
                groups.append((attempt, preparation + [node]))
                preparation = []
            else:
                groups[-1][1].append(node)
        elif groups:
            groups[-1][1].append(node)
        else:
            preparation.append(node)
    if preparation:
        groups.append((None, preparation))
    html = ["""<style>
    .flow{font:13px system-ui;color:#25334a;max-height:340px;overflow-y:auto;padding:0 2px}
    .flow-group+.flow-group{border-top:1px solid #e1e7ef;margin-top:10px;padding-top:10px}
    .flow-attempt{font-size:12px;font-weight:600;color:#64748b;margin-bottom:8px}
    .flow-scroll{overflow-x:auto;padding:0 1px 5px}
    .flow-track{display:flex;align-items:stretch;min-width:min-content}
    .flow-node{box-sizing:border-box;flex:1 0 108px;min-width:108px;border:1px solid #dce2eb;border-top:3px solid #c2cad7;border-radius:10px;padding:10px;background:#f6f8fb}
    .flow-node.running{background:#eaf2ff;border-color:#3984ef;color:#175bb5}
    .flow-node.success{background:#edf8f1;border-color:#3ba46b;color:#216942}
    .flow-node.failed{background:#fff0ef;border-color:#dd6a65;color:#a52c2c}
    .flow-node summary{list-style:none;cursor:pointer}
    .flow-node summary::-webkit-details-marker{display:none}
    .flow-node summary:focus-visible{outline:2px solid #2878e3;outline-offset:4px}
    .flow-node small{display:block;font-size:11px;line-height:1.6;margin-top:5px;color:inherit;opacity:.85}
    .flow-node p{font-size:12px;line-height:1.6;margin:8px 0 0;overflow-wrap:anywhere;max-width:260px}
    .flow-disclosure{float:right;opacity:.65;margin-left:4px}
    .flow-arrow{display:flex;flex:0 0 22px;align-items:center;justify-content:center;color:#94a3b8}
    </style><div class="flow" aria-label="本轮处理流程">"""]
    status_labels = dict(running="进行中", success="成功", failed="未通过", pending="等待", skipped="已跳过")
    icons = dict(running="●", success="✓", failed="×", pending="○", skipped="—")
    for attempt, nodes in groups:
        heading = f"第 {int(attempt)} 次尝试" if attempt is not None else "本轮处理"
        generation = next((n["title"] for n in nodes if n["id"].endswith(":generate")), None)
        if generation:
            heading += " · " + generation
        html.append(f'<section class="flow-group" data-attempt="{attempt if attempt is not None else "prepare"}">'
                    f'<div class="flow-attempt">{escape(heading)}</div><div class="flow-scroll"><div class="flow-track">')
        for index, node in enumerate(nodes):
            if index:
                html.append('<div class="flow-arrow" aria-hidden="true">→</div>')
            status = node["status"] if node["status"] in status_labels else "pending"
            label = "需复核" if node.get("needs_review") else status_labels[status]
            reason = REASONS.get(node.get("reason"), node.get("reason", ""))
            attributes = (f'class="flow-node {status}" data-node="{escape(node["id"], quote=True)}" '
                          f'data-state="{status}"')
            content = f'{escape(node["title"])}<small>{icons[status]} {label}</small>'
            if reason:
                html.append(f'<details {attributes}><summary><span class="flow-disclosure">⌄</span>{content}</summary>'
                            f'<p>{escape(str(reason))}</p></details>')
            else:
                html.append(f'<div {attributes}>{content}</div>')
        html.append('</div></div></section>')
    html.append('</div>')
    return "".join(html)


def preview_image(path, objects=()):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in objects:
        if "box" not in item:
            continue
        x1, y1, x2, y2 = item["box"]
        rect = [x1*image.width, y1*image.height, x2*image.width, y2*image.height]
        draw.rectangle(rect, outline="#3484ed", width=max(2, image.width//240))
        draw.text((rect[0]+3, rect[1]+3), str(item["class"]), fill="#174b9b", stroke_width=1, stroke_fill="white")
    return image


def prompt_text(directory, candidate, snapshot):
    if candidate.get("source") != "image":
        if snapshot.get("local_repair"):
            return "专用工具处理选区及周围 32 像素上下文，最终仅回贴选区。"
        return "专用工具直接处理原图。"
    for request in snapshot.get("edit_requests", []):
        if request["candidate"] == candidate.get("path") or request["prompt_file"] == candidate.get("prompt_file"):
            return request["prompt"]
    path = Path(directory) / candidate.get("prompt_file", "missing.txt")
    return path.read_text(encoding="utf-8") if path.is_file() else ""
