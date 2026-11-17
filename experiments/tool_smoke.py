import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from agentrestoration.core import Settings,Job,prepare,ROOT,write_json
from agentrestoration.adapters import Restormer
sys.path.insert(0,str(ROOT))
from experiments.runner import metrics
import torch
torch.set_num_threads(4)
s=Settings.load(ROOT / "configs/default.json")
job=Job(ROOT / "experiments" / "runs",s)
original=prepare(Path(r"F:\cvlab\datasets\test-data\denoising_testsets\CBSD68_25\101085.png"),job,s)
job.reserve("tool")
result=Restormer(s).run(original,job)
job.data.update(status="tool_inference_completed_unvalidated",output=result.name,
    metrics=metrics(result,r"F:\cvlab\datasets\test-data\denoising_testsets\CBSD68\101085.png"))
job.save()
write_json(ROOT / "experiments/tool_smoke.json",dict(directory=str(job.path),metrics=job.data["metrics"]))
print(job.path,job.data["metrics"])
