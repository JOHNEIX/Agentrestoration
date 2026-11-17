"""Thin UI-to-system bridge."""
from pathlib import Path
import time

from agentrestoration.adapters import Detector, Qwen
from agentrestoration.restoration_tools import RestorationTools
from agentrestoration.core import ROOT, Job, Settings, error_details
from agentrestoration.pipeline import run


def execute(path, requirement, task, on_event):
    job = Job(ROOT / "runs", Settings(), on_event=on_event, log_root=ROOT / "log/tasks")
    started = time.monotonic()
    job.data.update(requirement=requirement, requested_task=task)
    job.step("initialization", "初始化", "running")
    try:
        settings = Settings.load(ROOT / "configs/default.json")
        settings = Settings.model_validate(dict(settings.model_dump(), restoration_task=task))
        job.data["config"] = settings.model_dump()
        class LazyDetector:
            instance = None
            def detect(self, image):
                if self.instance is None:
                    self.instance = Detector(settings)
                return self.instance.detect(image)
        api, detector, tool = Qwen(settings), LazyDetector(), RestorationTools(settings)
        job.step("initialization", "初始化", "success")
    except Exception as error:
        details = error_details(error)
        job.data.update(status="execution_failed", error=details["message"], error_details=details,
                        elapsed_seconds=time.monotonic()-started)
        job.step("initialization", "初始化", "failed", reason=details["message"])
        job.event("execution_failed", error=details["message"], error_details=details)
        job.step("result", "最终结果", "failed", reason=details["message"])
        return job
    return run(Path(path), requirement, settings, api, detector, tool,
               output=ROOT / "runs", on_event=on_event, job=job)
