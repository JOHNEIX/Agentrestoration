"""Candidate presentation and validated parent links for user-initiated rounds."""
import json
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from agentrestoration.core import digest


def input_path(snapshot):
    if snapshot.get("directory"):
        normalized = Path(snapshot["directory"]) / "original_normalized.png"
        if normalized.is_file():
            return normalized
    source = Path(snapshot["source_file"]) if snapshot.get("source_file") else None
    return source if source and source.is_file() else None


def available_choices(snapshot):
    directory = Path(snapshot["directory"]) if snapshot.get("directory") else None
    result = []
    if directory:
        for candidate in snapshot.get("candidates", []):
            name = candidate.get("path")
            if not name:
                continue
            path = (directory/name).resolve()
            if path.parent != directory.resolve() or not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                continue
            result.append(dict(candidate, absolute_path=path))
    if not result:
        original = input_path(snapshot)
        if original:
            result.append(dict(id="original", label="原图", status="original",
                               source="original", absolute_path=original))
    return result


def chosen_option(snapshot, selected_id=None):
    choices = available_choices(snapshot)
    if not choices:
        return None
    explicit = next((c for c in choices if c["id"] == selected_id), None)
    return explicit or next((c for c in reversed(choices) if c["status"] == "accepted"), choices[-1])


class RoundOrigin(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    number: int = Field(ge=1)
    parent_task_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,100}$")
    parent_candidate_id: str | None = Field(default=None, min_length=1, max_length=100)
    initial_task_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,100}$")

    @model_validator(mode="after")
    def parent_pair(self):
        if bool(self.parent_task_id) != bool(self.parent_candidate_id):
            raise ValueError("上一轮任务与所选图片必须一起提供")
        if (self.number == 1) != (self.parent_task_id is None):
            raise ValueError("后续轮次必须关联所选图片")
        return self


def attach_origin(job, source, origin, root):
    context = RoundOrigin.model_validate(origin or dict(session_id=job.path.name, number=1))
    data = context.model_dump()
    if context.parent_task_id:
        record = root/"log/tasks"/context.parent_task_id/"record.log"
        parent = json.loads(record.read_text(encoding="utf-8"))
        parent_snapshot = dict(parent, directory=parent["artifact_directory"])
        choices = available_choices(parent_snapshot)
        # The original remains an explicit continuation source even when candidates exist.
        if context.parent_candidate_id == "original":
            expected = input_path(parent_snapshot)
        else:
            selected = next((c for c in choices if c["id"] == context.parent_candidate_id), None)
            expected = selected["absolute_path"] if selected else None
        if expected is None or Path(source).resolve() != expected.resolve():
            raise ValueError("下一轮输入与所选图片不一致")
        parent_context = parent.get("round", {})
        if parent_context.get("session_id", context.session_id) != context.session_id:
            raise ValueError("所选图片来自其他修复会话")
        if context.number <= parent_context.get("number", 0):
            raise ValueError("新轮次必须晚于来源轮次")
        initial = parent_context.get("initial_task_id", context.parent_task_id)
        if context.initial_task_id and context.initial_task_id != initial:
            raise ValueError("最初任务关联不一致")
        data.update(initial_task_id=initial, parent_record_file=str(record.resolve()),
                    selected_source_sha256=digest(expected))
    else:
        data["initial_task_id"] = job.path.name
    job.data["round"] = data
    job.event("round_started", **data)
