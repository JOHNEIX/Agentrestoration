import shutil
import time

from .adapters import APIError
from .core import ROOT, Job, choose_route, error_details, prepare, structure, write_json


def run(path, requirement, settings, api, detector, tool, output="runs", on_event=None,
        *, log_root=ROOT / "log/tasks", job=None):
    if job is None:
        job = Job(output, settings, on_event=on_event, log_root=log_root)
    job.data.update(requirement=requirement, provenance=getattr(api, "provenance", "real"), status="processing")
    started = time.monotonic()
    current = ("prepare", "图片准备")

    def begin(node_id, title, **details):
        nonlocal current
        current = (node_id, title)
        job.step(node_id, title, "running", **details)

    def end(status="success", **details):
        job.step(*current, status, **details)

    try:
        begin("prepare", "图片准备")
        original = prepare(path, job, settings)
        end()
        begin("analysis", "退化分析")
        analysis = api.analyze(original, requirement, job)
        job.data["analysis"] = analysis.model_dump()
        write_json(job.path / "analysis.json", analysis.model_dump())
        end()
        prompt, route = analysis.edit_prompt, "image"
        attempt = 0
        while route != "stop":
            attempt += 1
            number = job.data["image_requests_sent"] + 1 if route == "image" else job.data["tool_pipeline_started"] + 1
            candidate_id = f"{route}_{number}"
            label = ("首次编辑" if number == 1 else "编辑重试") if route == "image" else "工具回退"
            candidate_record = dict(id=candidate_id, attempt=attempt, label=label, source=route, status="executing")
            if route == "tool":
                candidate_record["tool_name"] = getattr(tool, "name", "专用工具")
                label = candidate_record["label"] = "工具回退 · " + candidate_record["tool_name"]
            if route == "image":
                candidate_record["prompt_file"] = f"prompts/image_{number}.txt"
            job.data["candidates"].append(candidate_record)
            for phase, title in (("generate", label), ("structure", "结构检查"), ("quality", "效果验证")):
                job.step(f"{candidate_id}:{phase}", title, "pending", attempt=attempt, source=route)
            severe, feedback = False, ""
            images_before = job.data["image_requests_sent"]
            try:
                begin(f"{candidate_id}:generate", label, attempt=attempt, source=route)
                if route == "image":
                    candidate = api.edit(original, prompt, job)
                else:
                    job.reserve("tool")
                    candidate = tool.run(original, job)
                candidate_record["path"] = candidate.name
                end(candidate_path=candidate.name)
                job.event("candidate_ready", candidate_id=candidate_id, path=candidate.name, attempt=attempt)

                begin(f"{candidate_id}:structure", "结构检查", attempt=attempt)
                baseline_uncertain = any(d.uncertain or (d.type in ("blur", "low_light") and d.severity in ("moderate", "severe", "uncertain"))
                                         for d in analysis.degradations)
                checks = structure(original, candidate, detector, settings, baseline_uncertain)
                candidate_record["checks"] = checks
                end("success" if checks["status"] == "pass" else "failed",
                    reason=checks["reason"], needs_review=checks["status"] == "uncertain")

                if checks["status"] == "fail":
                    candidate_record.update(status="rejected", reason=checks["reason"])
                    severe, feedback = checks["severe"], str(checks)
                    job.step(f"{candidate_id}:quality", "效果验证", "skipped", attempt=attempt, reason="结构异常，转入路线选择")
                else:
                    begin(f"{candidate_id}:quality", "效果验证", attempt=attempt)
                    verdict = api.compare(original, candidate, checks, job)
                    candidate_record["verdict"] = verdict.model_dump()
                    candidate_record["status"] = "accepted" if verdict.accepted() else "rejected"
                    end("success" if verdict.accepted() else "failed", reason=verdict.reason,
                        needs_review=verdict.decision == "uncertain")
                    if verdict.accepted():
                        shutil.copyfile(candidate, job.path / "final.png")
                        job.data.update(status="completed", accepted=True, output="final.png")
                        break
                    feedback = verdict.reason + " " + verdict.retry_instruction
                    severe = verdict.structure_preserved is False and verdict.structure_issue == "count_change"
                prompt = analysis.edit_prompt + "\nCorrect previous failure: " + feedback + "\nAlways preserve original content and composition."
            except Exception as error:
                details = error_details(error)
                message = str(error) if isinstance(error, APIError) else details["message"]
                candidate_record.update(status="execution_failed", error=message, error_details=details)
                end("failed", reason=message)
                for phase in ("structure", "quality"):
                    node = next(s for s in job.data["steps"] if s["id"] == f"{candidate_id}:{phase}")
                    if node["status"] == "pending":
                        job.step(node["id"], node["title"], "skipped", attempt=attempt, reason="前序阶段执行失败")
                job.event("execution_failed", error=message, error_details=details)
                if route == "image" and job.data["image_requests_sent"] == images_before:
                    break

            begin(f"route_{attempt}", "路线选择", attempt=attempt)
            available = tool.available(analysis)
            route = choose_route(severe, available, job.data["image_requests_sent"], job.data["tool_pipeline_started"])
            reasons = {"image": "根据问题修订指令，使用剩余编辑额度", "tool": "执行适用的专用工具", "stop": "可用路线已用尽"}
            end(route=route, reason=reasons[route], severe=severe, tool_available=available)
            job.event("routing", route=route, severe=severe, tool_available=available)
        if job.data["status"] != "completed":
            has_rejection = any(c["status"] == "rejected" for c in job.data["candidates"])
            job.data.update(status="needs_review" if has_rejection else "execution_failed", output="original_normalized.png")
            if not has_rejection:
                job.data["error"] = next((c["error"] for c in reversed(job.data["candidates"]) if c.get("error")), "execution_failed")
    except Exception as error:
        details = error_details(error)
        message = str(error) if isinstance(error, APIError) else details["message"]
        job.data.update(status="execution_failed", error=message, error_details=details)
        end("failed", reason=message)
        job.event("execution_failed", error=message, error_details=details)
    finally:
        job.data["elapsed_seconds"] = time.monotonic() - started
        reason = {"completed": "修复结果已接受", "needs_review": "保留原图，候选可供复核", "execution_failed": "处理执行失败"}[job.data["status"]]
        if job.data["status"] == "execution_failed" and job.data.get("error"):
            reason += ": " + job.data["error"]
        job.step("result", "最终结果", "success" if job.data["accepted"] else "failed", reason=reason)
    return job
