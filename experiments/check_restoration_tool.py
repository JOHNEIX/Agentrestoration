"""Run one real tool inference and record assets, dimensions and timing."""
import argparse
import sys
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentrestoration.core import ROOT, Analysis, Job, Settings, digest, prepare
from agentrestoration.restoration_tools import RestorationTools
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--task", choices=["low_light", "dehazing", "gaussian_denoising"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/runs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    args = parser.parse_args()
    settings = Settings.load(args.config).model_copy(update={"restoration_task": args.task})
    kind = {"low_light": "low_light", "dehazing": "haze", "gaussian_denoising": "noise"}[args.task]
    tools = RestorationTools(settings)
    analysis = Analysis(degradations=[dict(type=kind, severity="moderate", uncertain=False)],
                        restoration_goal=args.task, preserve=[], edit_prompt=args.task)
    if not tools.available(analysis):
        parser.error("Tool assets unavailable")
    job = Job(args.output, settings)
    original = prepare(args.image, job, settings)
    job.reserve("tool")
    start = time.monotonic()
    candidate = tools.run(original, job)
    with Image.open(original) as source, Image.open(candidate) as result:
        assert source.size == result.size
        size = list(result.size)
    job.data.update(status="tool_inference_completed_unvalidated", output=candidate.name,
                    tool_name=tools.name, input_sha256=digest(original), output_sha256=digest(candidate),
                    dimensions=size, elapsed_seconds=time.monotonic()-start, evidence="real_author_weights")
    job.save()
    print(job.path, flush=True)
    print(job.data["tool_name"], size, round(job.data["elapsed_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
