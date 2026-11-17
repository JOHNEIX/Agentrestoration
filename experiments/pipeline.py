"""Experimental baselines; the system always runs the full restoration loop."""
import time
from agentrestoration.core import Analysis, Job, prepare, write_json
from agentrestoration.adapters import APIError
from agentrestoration.pipeline import run as restore

DEFAULT_PROMPT = "Restore image quality conservatively: remove visible noise, improve clarity, preserve all original objects, positions, composition, aspect ratio and colors. Do not invent or redraw content."

class NoFallback:
    def available(self, analysis):
        return False

def run(path, requirement, settings, api, detector, tool, output, group):
    if group not in ("A", "B", "C", "D"):
        raise ValueError("Invalid experiment group")
    if group in ("C", "D"):
        job = restore(path, requirement, settings, api, detector,
                      tool if group == "D" else NoFallback(), output, log_root=None)
        job.data["group"] = group
        job.save()
        return job
    job = Job(output, settings)
    job.data.update(group=group, requirement=requirement,
                    provenance=getattr(api, "provenance", "real"))
    started = time.monotonic()
    try:
        original = prepare(path, job, settings)
        analysis = (Analysis(degradations=[], restoration_goal=requirement,
                             preserve=[], edit_prompt=DEFAULT_PROMPT)
                    if group == "A" else api.analyze(original, requirement, job))
        write_json(job.path / "analysis.json", analysis.model_dump())
        while job.data["image_requests_sent"] < 2:
            before = job.data["image_requests_sent"]
            candidate = {"source": "image", "status": "executing"}
            job.data["candidates"].append(candidate)
            try:
                image = api.edit(original, analysis.edit_prompt, job)
                candidate.update(path=image.name, status="unvalidated")
                job.data.update(status="baseline_completed", accepted=None, output=image.name)
                break
            except Exception as error:
                candidate.update(status="execution_failed",
                                 error=str(error) if isinstance(error, APIError) else type(error).__name__)
                job.event("execution_failed", error=candidate["error"])
                if job.data["image_requests_sent"] == before:
                    break
        if job.data["status"] != "baseline_completed":
            job.data.update(status="execution_failed", output="original_normalized.png")
    except Exception as error:
        job.data.update(status="execution_failed",
                        error=str(error) if isinstance(error, APIError) else type(error).__name__)
    finally:
        job.data["elapsed_seconds"] = time.monotonic() - started
        job.save()
    return job
