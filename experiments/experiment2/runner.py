"""Execute A/B/C with real adapters; system code has no experiment switches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

from agentrestoration.adapters import Detector, Qwen
from agentrestoration.core import ROOT, Job, Settings, digest, error_details, prepare, write_json
from agentrestoration.pipeline import run
from agentrestoration.restoration_tools import RestorationTools
from .protocol import evaluate, make_manifest, pixels
from .shared import NoTool, SerialTool, SharedAPI

REQUIREMENT = "去除图像中的高斯噪声，保留原始物体、细节、颜色、背景、构图和物体位置。"


class LockedDetector:
    def __init__(self, detector):
        self.detector, self.lock = detector, threading.Lock()

    def detect(self, path):
        with self.lock:
            return self.detector.detect(path)


def group_result(job, scene, repeat, group, destination):
    output = job.path/job.data["output"]
    metrics = evaluate(output,scene["reference"],destination/job.path.name/"evaluation.png",
                       job.data["config"]["ratio_pass"])
    baseline = scene["input_metrics"]
    actual_pixels, input_pixels = pixels(output), pixels(scene["input"])
    unchanged = actual_pixels.size == input_pixels.size and actual_pixels.tobytes() == input_pixels.tobytes()
    metrics["delta_psnr"] = (metrics["psnr"]-baseline["psnr"]
                              if metrics["psnr"] is not None and baseline["psnr"] is not None else None)
    metrics["delta_ssim"] = metrics["ssim"]-baseline["ssim"] if metrics["ssim"] is not None else None
    events = job.data["events"]
    accounting = job.data.get("experiment", {})
    actual = {kind:sum(e["stage"] == "api_request" and e.get("kind") == kind for e in events)
              for kind in ("image","vlm")}
    return dict(scene=scene["scene"], repeat=repeat, group=group, status=job.data["status"],
                accepted=job.data["accepted"], output=str(output.resolve()), output_sha256=digest(output),
                record_file=str(job.record_path.resolve()), metrics=metrics, input_metrics=baseline,
                equivalent_image_requests=job.data["image_requests_sent"],
                actual_image_requests=actual["image"], actual_vlm_requests=actual["vlm"],
                equivalent_vlm_requests=actual["vlm"]+accounting.get("reused_vlm_requests",0),
                tool_runs=job.data["tool_pipeline_started"],
                actual_seconds=job.data["elapsed_seconds"],
                equivalent_seconds=job.data["elapsed_seconds"]+accounting.get("timing_credit_seconds",0)
                -accounting.get("tool_queue_seconds",0),
                execution_errors=sum(e["stage"] == "execution_failed" for e in events),
                output_is_input=unchanged,
                first_candidate_sha256=accounting.get("first_candidate_sha256"))


def recoverable_trial(trial):
    """Only recover when every image request is cached and no tool was started."""
    entries = json.loads((trial/"shared_operations.json").read_text(encoding="utf-8"))
    actual_images, tools = 0, 0
    previous = []
    for folder in trial.glob("[ABC]/*"):
        events = []
        event_file = folder/"events.jsonl"
        if event_file.exists():
            for line in event_file.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        actual_images += sum(e["stage"]=="api_request" and e.get("kind")=="image" for e in events)
        tools += sum(e["stage"]=="budget_reserved" and e.get("source")=="tool" for e in events)
        previous.append(str(folder.resolve()))
    cached_images = sum(r["kind"]=="image" for e in entries.values()
                        if e["operation"]=="edit" for r in e["requests"])
    if actual_images != cached_images or tools:
        raise RuntimeError("Unknown image/tool outcome; interrupted trial is retained for review")
    for entry in entries.values():
        if entry["operation"]=="edit" and "error" not in entry:
            if not Path(entry["value"]).is_file():
                raise RuntimeError("Cached candidate is missing")
    write_json(trial/"recovery.json",dict(previous_jobs=previous,
        time=datetime.now(timezone.utc).isoformat(),cached_image_requests=cached_images,
        image_requests_reissued=0,reason="Explicit continuation after session interruption"))


def run_trial(scene, repeat, settings, trial, detector, tool_lock, api=None, tool=None, recover=False):
    trial = Path(trial)
    trial.mkdir(parents=True,exist_ok=True)
    result_file = trial/"result.json"
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    if (trial/"started.json").exists():
        if not recover:
            raise RuntimeError("Interrupted trial is retained without automatic request replay: "+str(trial))
        recoverable_trial(trial)
    write_json(trial/"started.json",dict(scene=scene["scene"],repeat=repeat,
                                       time=datetime.now(timezone.utc).isoformat()))
    real_api = api or Qwen(settings)
    shared = SharedAPI(real_api,trial)
    rows = []
    jobs = []
    started = time.monotonic()
    try:
        first = Job(trial/"A",settings)
        jobs.append(first)
        first.data.update(requirement=REQUIREMENT, provenance=shared.provenance,
                          experiment=dict(group="A",scene=scene["scene"],repeat=repeat))
        first.step("prepare","图片准备","running")
        original = prepare(scene["input"],first,settings)
        first.step("prepare","图片准备","success")
        first.step("analysis","退化分析","running")
        analysis = shared.analyze(original,REQUIREMENT,first)
        first.data["analysis"] = analysis.model_dump()
        write_json(first.path/"analysis.json",analysis.model_dump())
        first.step("analysis","退化分析","success")
        first.step("image_1:generate","首次编辑","running",attempt=1)
        candidate = shared.edit(original,analysis.edit_prompt,first)
        first_hash = digest(candidate)
        first.data["candidates"].append(dict(id="image_1",path=candidate.name,source="image",
                                             status="unvalidated",prompt_file="prompts/image_1.txt"))
        first.data.update(status="baseline_completed",accepted=None,output=candidate.name,
                          elapsed_seconds=time.monotonic()-started)
        first.data["experiment"]["first_candidate_sha256"] = first_hash
        first.step("image_1:generate","首次编辑","success",attempt=1)
        rows.append(group_result(first,scene,repeat,"A",trial/"evaluation"))
        for group in ("B","C"):
            job = Job(trial/group,settings)
            jobs.append(job)
            job.data["experiment"] = dict(group=group,scene=scene["scene"],repeat=repeat,
                                         first_candidate_sha256=first_hash)
            selected_tool = NoTool() if group == "B" else SerialTool(
                tool or RestorationTools(settings),tool_lock)
            run(scene["input"],REQUIREMENT,settings,shared,detector,selected_tool,job=job)
            first_output = job.path/job.data["candidates"][0]["path"]
            if digest(first_output) != first_hash:
                raise RuntimeError("First candidate differs between routes")
            assert job.data["image_requests_sent"] <= 2 and job.data["tool_pipeline_started"] <= 1
            if job.data["status"] == "execution_failed":
                raise RuntimeError(group+" execution failed: "+job.data.get("error","unknown"))
            rows.append(group_result(job,scene,repeat,group,trial/"evaluation"))
        result = dict(status="completed",scene=scene["scene"],repeat=repeat,
                      first_candidate_sha256=first_hash,rows=rows,
                      actual_wall_seconds=time.monotonic()-started)
    except Exception as error:
        details = error_details(error)
        # APIError text is already redacted by the production adapter.
        from agentrestoration.adapters import APIError
        message = str(error) if isinstance(error,APIError) else details["message"]
        if jobs and jobs[-1].data["status"] in ("preparing","processing"):
            jobs[-1].data.update(status="execution_failed",error=message,error_details=details)
            jobs[-1].event("execution_failed",error=message,error_details=details)
        result = dict(status="execution_failed",scene=scene["scene"],repeat=repeat,
                      error=message,error_details=details,rows=rows,
                      actual_wall_seconds=time.monotonic()-started)
    finally:
        if api is None:
            real_api.client.close()
    write_json(result_file,result)
    return result


def source_hashes():
    sources = list((ROOT/"src/agentrestoration").glob("*.py"))
    sources += list((ROOT/"experiments/experiment2").glob("*.py"))
    sources.append(ROOT/"experiments/run_experiment2.py")
    if (ROOT/"experiments/rerun_experiment2.py").exists():
        sources.append(ROOT/"experiments/rerun_experiment2.py")
    return {str(p.relative_to(ROOT)).replace("\\","/"):digest(p) for p in sources}


def prepare_run(args):
    settings = Settings.load(ROOT/args.config)
    settings.restoration_task = "gaussian_denoising"
    directory = ROOT/"experiments/runs"/(
        "experiment2_"+datetime.now().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    scenes = make_manifest(args.data_root,args.scenes,args.seed,directory)
    protocol = dict(experiment="2",created_at=datetime.now(timezone.utc).isoformat(),
                    scenes=args.scenes,repeats=args.repeats,seed=args.seed,workers=args.workers,
                    dataset="CBSD68_25",settings=settings.model_dump(),requirement=REQUIREMENT,
                    source_sha256=source_hashes(),evaluation=dict(data_range=255,channel_axis=2,
                    ssim_win_size=7,ssim_gaussian_weights=False,ssim_sample_covariance=True,
                    ratio_limit=settings.ratio_pass,mapping="LANCZOS_to_reference_after_ratio_gate"),
                    sharing="Same analysis, first candidate and identical prompt/validation operations per trial",
                    gallery_selection="Top two and bottom one trial by paired C-B PSNR difference",
                    timing="Measured wall time plus reused operation duration minus replay and tool queue time",
                    generation_seed="not set; seed applies to scene selection",
                    packages={p:version(p) for p in ("Pillow","numpy","scikit-image","matplotlib","httpx")})
    write_json(directory/"protocol.json",protocol)
    write_json(directory/"manifest.json",scenes)
    for name in protocol["source_sha256"]:
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return directory,protocol,scenes


def collect(directory):
    results = []
    for path in sorted((directory/"trials").glob("*/*/result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(directory/"results.json",results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,default=Path(r"F:\cvlab\datasets\test-data\denoising_testsets"))
    parser.add_argument("--config",default="configs/default.json")
    parser.add_argument("--scenes",type=int,default=20)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20261112)
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=2)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--report-only",type=Path)
    parser.add_argument("--recover-interrupted",action="store_true")
    args = parser.parse_args()
    if args.report_only:
        from .report import build_report
        build_report(args.report_only.resolve())
        return
    if args.scenes <= 0 or args.repeats <= 0:
        parser.error("scenes and repeats must be positive")
    if args.resume:
        directory = args.resume.resolve()
        protocol = json.loads((directory/"protocol.json").read_text(encoding="utf-8"))
        scenes = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        revisions_file = directory/"source_revisions.json"
        revisions = json.loads(revisions_file.read_text(encoding="utf-8")) if revisions_file.exists() else []
        previous_hashes = revisions[-1]["source_sha256"] if revisions else protocol["source_sha256"]
        current_hashes = source_hashes()
        if previous_hashes != current_hashes:
            if not args.recover_interrupted or any(
                    current_hashes.get(k) != v for k,v in protocol["source_sha256"].items()
                    if k.startswith("src/")):
                raise RuntimeError("Source differs from the frozen protocol")
            revisions.append(dict(time=datetime.now(timezone.utc).isoformat(),
                                  source_sha256=current_hashes,reason="Interrupted-trial recovery support"))
            write_json(revisions_file,revisions)
            for name in current_hashes:
                target = directory/f"recovery_source_{len(revisions)}"/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
    else:
        directory,protocol,scenes = prepare_run(args)
    print("EXPERIMENT_DIRECTORY="+str(directory),flush=True)
    if args.prepare_only:
        print("Protocol and paired scenes saved; no API requests sent.",flush=True)
        return
    settings = Settings.model_validate(protocol["settings"])
    from agentrestoration.adapters import credentials
    if not all(credentials().values()):
        raise RuntimeError("API credentials are not configured")
    detector = LockedDetector(Detector(settings))
    detector.detect(scenes[0]["input"])  # Model warmup is outside group timings.
    tool_lock = threading.Lock()
    pending = [(s,r) for s in scenes for r in range(1,protocol["repeats"]+1)]
    pending = [(s,r) for s,r in pending if not
               (directory/"trials"/s["scene"]/f"repeat_{r:02d}"/"result.json").exists()]
    started = time.monotonic()
    stop = False
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as executor:
        active = {}
        while pending or active:
            while pending and len(active) < protocol["workers"] and not stop:
                scene,repeat = pending.pop(0)
                task = executor.submit(run_trial,scene,repeat,settings,
                                       directory/"trials"/scene["scene"]/f"repeat_{repeat:02d}",
                                       detector,tool_lock,recover=args.recover_interrupted)
                active[task] = (scene["scene"],repeat)
            if not active:
                break
            done,_ = wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                scene,repeat = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    stop = True
                    print(f"STOP {scene} repeat={repeat} {type(error).__name__}",flush=True)
                    continue
                all_results = collect(directory)
                done_count = sum(r["status"] == "completed" for r in all_results)
                detail = " ".join(f"{r['group']}:{r['status']}" for r in result["rows"])
                print(f"[{done_count}/{protocol['scenes']*protocol['repeats']}] {scene} repeat={repeat} {detail}",flush=True)
                if result["status"] != "completed":
                    stop = True
                    print("STOP "+result["error"],flush=True)
            write_json(directory/"progress.json",dict(
                completed=sum(r["status"]=="completed" for r in collect(directory)),
                planned=protocol["scenes"]*protocol["repeats"],stopped=stop,
                current_session_seconds=time.monotonic()-started))
    from .report import build_report
    build_report(directory)
    print("FINISHED "+str(directory/"report.md"),flush=True)
    if stop:
        raise SystemExit(2)
