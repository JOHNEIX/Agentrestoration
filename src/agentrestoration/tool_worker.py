"""Isolated inference worker. Parent alone owns the budget."""
import sys
from pathlib import Path
from .core import Settings,write_json
from .adapters import Restormer

class WorkerJob:
    def __init__(self, directory): self.path=Path(directory)
    def event(self, stage, **metadata): write_json(self.path / "tool_metadata.json",metadata)

if __name__=="__main__":
    try:
        Restormer(Settings.load(sys.argv[3])).infer(Path(sys.argv[1]),WorkerJob(sys.argv[2]))
    except Exception as e:
        text=str(e).lower()
        kind="out_of_memory" if "out of memory" in text else "weights_mismatch" if "state_dict" in text else type(e).__name__
        write_json(Path(sys.argv[2])/"tool_error.json",dict(kind=kind))
        raise SystemExit(1)
