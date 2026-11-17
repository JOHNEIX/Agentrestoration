import argparse
import json
from .core import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--requirement", default="改善图像质量，尽量保留原始内容、构图和物体位置")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--task", choices=["auto", "gaussian_denoising", "low_light", "dehazing"], default=None)
    args = parser.parse_args()
    from .adapters import Qwen, Detector
    from .restoration_tools import RestorationTools
    from .pipeline import run
    s = Settings.load(args.config)
    if args.task:
        s.restoration_task = args.task
    print("图片将发送至所配置的外部 API。")
    class LazyDetector:
        instance = None
        def detect(self, path):
            if self.instance is None:
                self.instance = Detector(s)
            return self.instance.detect(path)
    job = run(args.image, args.requirement, s, Qwen(s), LazyDetector(), RestorationTools(s), args.output)
    print(json.dumps(dict(status=job.data["status"], directory=str(job.path),
                          log_directory=str(job.log_path)), ensure_ascii=False))
    return 0 if job.data["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
