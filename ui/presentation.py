"""Pure presentation helpers, shared by the UI and its tests."""
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

LABELS = {"accepted": "已接受", "rejected": "未通过", "executing": "正在处理", "execution_failed": "执行失败"}
REASONS = {"aspect_ratio": "宽高比变化", "object_position": "目标位置变化", "matched": "主要目标匹配",
           "suspected_count_change": "目标对应关系需复核", "high_confidence_suspected_count_change": "疑似目标增删",
           "unreliable_or_truncated_detection": "目标检测需复核"}


def candidate_label(candidate):
    name = candidate.get("label") or ("工具回退" if candidate["source"] == "tool" else candidate.get("path", "编辑候选"))
    status = "需复核" if candidate.get("verdict", {}).get("decision") == "uncertain" else LABELS.get(candidate['status'], candidate['status'])
    return f"{name} · {status}"


def flow_html(snapshot):
    steps = snapshot.get("steps") or [dict(id="prepare", title="图片准备", status="pending"),
                                      dict(id="analysis", title="退化分析", status="pending"),
                                      dict(id="image_1:generate", title="首次编辑", status="pending")]
    html = ["""<style>
    .flow{font:14px system-ui;color:#25334a;padding:4px 2px;max-height:740px;overflow:auto}
    .flow-node{border:1px solid #dce2eb;border-left:4px solid #c2cad7;border-radius:11px;padding:12px 14px;background:#f6f8fb}
    .flow-node.running{background:#eaf2ff;border-color:#3984ef;color:#175bb5}
    .flow-node.success{background:#edf8f1;border-color:#3ba46b;color:#216942}
    .flow-node.failed{background:#fff0ef;border-color:#dd6a65;color:#a52c2c}
    .flow-node small{display:block;font-size:12px;line-height:1.5;margin-top:5px;overflow-wrap:anywhere;color:inherit;opacity:.85}
    .flow-arrow{text-align:center;color:#a6b2c3;height:22px;line-height:22px}
    .flow-attempt{font-size:12px;color:#607086;margin:16px 0 8px}
    </style><div class="flow">"""]
    previous_attempt = None
    status_labels = dict(running="进行中", success="成功", failed="未通过", pending="等待", skipped="已跳过")
    icons = dict(running="●", success="✓", failed="×", pending="○", skipped="—")
    for index, node in enumerate(steps):
        if index:
            html.append('<div class="flow-arrow">↓</div>')
        attempt = node.get("attempt")
        if attempt and attempt != previous_attempt:
            html.append(f'<div class="flow-attempt">第 {int(attempt)} 次尝试</div>')
            previous_attempt = attempt
        status = node["status"] if node["status"] in status_labels else "pending"
        label = "需复核" if node.get("needs_review") else status_labels[status]
        reason = REASONS.get(node.get("reason"), node.get("reason", ""))
        html.append(f'<div class="flow-node {status}" data-node="{escape(node["id"], quote=True)}" data-state="{status}">'
                    f'{icons[status]} {escape(node["title"])} · {label}'
                    + (f'<small>{escape(str(reason))}</small>' if reason else "") + '</div>')
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
        return "专用工具直接处理原图。"
    for request in snapshot.get("edit_requests", []):
        if request["candidate"] == candidate.get("path") or request["prompt_file"] == candidate.get("prompt_file"):
            return request["prompt"]
    path = Path(directory) / candidate.get("prompt_file", "missing.txt")
    return path.read_text(encoding="utf-8") if path.is_file() else ""
