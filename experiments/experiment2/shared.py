"""Reuse identical model operations within one scene/repetition."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from agentrestoration.adapters import APIError
from agentrestoration.core import Analysis, Verdict, digest, write_json


class SharedAPI:
    def __init__(self, api, directory):
        self.api = api
        self.provenance = getattr(api, "provenance", "real")
        self.directory = Path(directory)
        saved = self.directory/"shared_operations.json"
        self.entries = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}

    def _call(self, operation, inputs, job, invoke, schema=None, prompt=None):
        key = hashlib.sha256(json.dumps([operation, inputs], sort_keys=True,
                                       ensure_ascii=False).encode()).hexdigest()
        started = time.monotonic()
        if key not in self.entries:
            before = job.data["image_requests_sent"]
            event_count = len(job.data["events"])
            entry = dict(operation=operation, source_record=str(job.record_path.resolve()))
            try:
                result = invoke()
                entry["value"] = str(result.resolve()) if operation == "edit" else result.model_dump()
            except Exception as error:
                entry["error"] = str(error) if isinstance(error, APIError) else type(error).__name__
                raise
            finally:
                entry["seconds"] = time.monotonic()-started
                entry["reserved_image"] = job.data["image_requests_sent"] > before
                entry["requests"] = [dict(kind=e["kind"], model=e.get("model"))
                                     for e in job.data["events"][event_count:] if e["stage"] == "api_request"]
                if operation == "edit" and job.data["edit_requests"]:
                    entry["edit_request"] = deepcopy(job.data["edit_requests"][-1])
                self.entries[key] = entry
                write_json(self.directory/"shared_operations.json", self.entries)
            return result

        entry = self.entries[key]
        if operation == "edit":
            index = job.data["image_requests_sent"]+1
            if entry["reserved_image"]:
                job.reserve("image")
                filename = f"prompts/image_{index}.txt"
                (job.path/"prompts").mkdir(exist_ok=True)
                (job.path/filename).write_text(prompt, encoding="utf-8")
                request = deepcopy(entry.get("edit_request", {}))
                request.update(index=index, prompt=prompt, prompt_file=filename,
                               candidate=f"image_{index}.png", status="reused",
                               reused_from=entry["source_record"], actually_sent_here=False)
                job.data["edit_requests"].append(request)
            if "error" not in entry:
                result = job.path/f"image_{index}.png"
                shutil.copyfile(entry["value"], result)
        elif "error" not in entry:
            result = schema.model_validate(entry["value"])
        replay_seconds = time.monotonic()-started
        accounting = job.data.setdefault("experiment", {})
        accounting["timing_credit_seconds"] = accounting.get("timing_credit_seconds", 0) + max(
            0, entry["seconds"]-replay_seconds)
        accounting["reused_vlm_requests"] = accounting.get("reused_vlm_requests", 0) + sum(
            r["kind"] == "vlm" for r in entry["requests"])
        job.event("experiment_operation_reused", operation=operation, key=key,
                  source_record=entry["source_record"], original_seconds=entry["seconds"],
                  replay_seconds=replay_seconds, error=entry.get("error"))
        if "error" in entry:
            raise APIError(entry["error"])
        return result

    def analyze(self, original, requirement, job):
        return self._call("analyze", [digest(original), requirement], job,
                          lambda: self.api.analyze(original, requirement, job), Analysis)

    def edit(self, original, prompt, job):
        return self._call("edit", [digest(original), prompt], job,
                          lambda: self.api.edit(original, prompt, job), prompt=prompt)

    def compare(self, original, candidate, checks, job):
        return self._call("compare", [digest(original), digest(candidate), checks,
                                     job.data["requirement"]], job,
                          lambda: self.api.compare(original, candidate, checks, job), Verdict)


class NoTool:
    def available(self, analysis):
        return False


class SerialTool:
    def __init__(self, tool, lock):
        self.tool, self.lock = tool, lock

    @property
    def name(self):
        return self.tool.name

    def available(self, analysis):
        return self.tool.available(analysis)

    def run(self, original, job):
        waiting = time.monotonic()
        with self.lock:
            job.data.setdefault("experiment", {})["tool_queue_seconds"] = time.monotonic()-waiting
            return self.tool.run(original, job)
