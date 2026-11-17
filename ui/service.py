"""Thin UI-to-system bridge."""
from pathlib import Path

from agentrestoration.adapters import Detector, Qwen, Restormer
from agentrestoration.core import ROOT, Settings
from agentrestoration.pipeline import run


def execute(path, requirement, task, on_event):
    settings = Settings.load(ROOT / "configs/default.json")
    settings.restoration_task = task
    class LazyDetector:
        instance = None
        def detect(self, image):
            if self.instance is None:
                self.instance = Detector(settings)
            return self.instance.detect(image)
    return run(Path(path), requirement, settings, Qwen(settings), LazyDetector(), Restormer(settings),
               output=ROOT / "runs", on_event=on_event)
