"""Thin UI-to-system bridge."""
from pathlib import Path
import time
import signal
import threading

from agentrestoration.adapters import Detector, Qwen
from agentrestoration.restoration_tools import RestorationTools
from agentrestoration.core import ROOT, Job, Settings, error_details
from agentrestoration.pipeline import run
from .rounds import attach_origin


_lifecycle_lock = threading.Lock()
_active_jobs = 0
_shutdown_pending = False


def shutdown():
    """Use Streamlit's installed Ctrl+C handler to close the local server."""
    global _shutdown_pending
    with _lifecycle_lock:
        if _active_jobs:
            raise RuntimeError("正在修复图片，请等待处理完成后退出。")
        if _shutdown_pending:
            return
        handler = signal.getsignal(signal.SIGINT)
        if not callable(handler) or handler is signal.default_int_handler:
            raise RuntimeError("服务关闭接口未就绪，请在启动窗口按 Ctrl+C。")
        # Allow the message to reach the page before closing connections.
        timer = threading.Timer(.5, handler, args=(signal.SIGINT, None))
        timer.daemon = True
        _shutdown_pending = True
        try:
            timer.start()
        except Exception:
            _shutdown_pending = False
            raise


def execute(path, requirement, task, on_event, region=None, origin=None):
    global _active_jobs
    with _lifecycle_lock:
        if _shutdown_pending:
            raise RuntimeError("工作台正在退出，请重新启动后处理图片。")
        _active_jobs += 1
    try:
        options = {}
        if region is not None: options["region"] = region
        if origin is not None: options["origin"] = origin
        return _execute(path, requirement, task, on_event, **options)
    finally:
        with _lifecycle_lock:
            _active_jobs -= 1


def _execute(path, requirement, task, on_event, region=None, origin=None):
    job = Job(ROOT / "runs", Settings(), on_event=on_event, log_root=ROOT / "log/tasks")
    started = time.monotonic()
    job.data.update(requirement=requirement, requested_task=task,source_file=str(Path(path).resolve()))
    job.step("initialization", "初始化", "running")
    try:
        settings = Settings.load(ROOT / "configs/default.json")
        settings = Settings.model_validate(dict(settings.model_dump(), restoration_task=task))
        attach_origin(job,path,origin,ROOT)
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
               output=ROOT / "runs", on_event=on_event, job=job, region=region)
