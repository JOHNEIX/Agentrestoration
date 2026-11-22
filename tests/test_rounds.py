import json
from pathlib import Path

from PIL import Image
import pytest

from agentrestoration.core import digest
from ui.rounds import available_choices, chosen_option
from ui.presentation import candidate_label
from ui import service
from test_core import FakeAPI, Detection, Tool


def snapshot_with_images(tmp_path, statuses):
    Image.new("RGB", (64, 64), "gray").save(tmp_path/"original_normalized.png")
    candidates = []
    for i, status in enumerate(statuses, 1):
        path = f"image_{i}.png"
        Image.new("RGB", (64, 64), (100+i,)*3).save(tmp_path/path)
        candidates.append(dict(id=f"image_{i}", path=path, source="image", status=status,
                               verdict=dict(reason=f"reason {i}")))
    return dict(directory=str(tmp_path), candidates=candidates)


def test_candidate_defaults_selection_and_invalid_images(tmp_path):
    snapshot = snapshot_with_images(tmp_path, ["rejected", "accepted", "rejected"])
    assert chosen_option(snapshot)["id"] == "image_2"
    assert chosen_option(snapshot, "image_1")["status"] == "rejected"
    snapshot["candidates"][1]["status"] = "rejected"
    assert chosen_option(snapshot)["id"] == "image_3"
    (tmp_path/"image_3.png").write_bytes(b"not an image")
    assert chosen_option(snapshot)["id"] == "image_2"
    snapshot["candidates"].append(dict(id="escape",path="../outside.png"))
    assert len(available_choices(snapshot)) == 2
    snapshot["candidates"] = []
    assert chosen_option(snapshot)["id"] == "original"


def prepare_service(tmp_path, monkeypatch, api_factory):
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/default.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "Qwen", lambda settings: api_factory())
    monkeypatch.setattr(service, "Detector", lambda settings: Detection())
    monkeypatch.setattr(service, "RestorationTools", lambda settings: Tool())
    return lambda path, origin=None: service.execute(str(path), "denoise", "auto", None, origin=origin)


def test_round_lineage_independent_budgets_and_unchanged_parent(tmp_path, monkeypatch):
    execute = prepare_service(tmp_path, monkeypatch, FakeAPI)
    source = tmp_path/"input.png"
    Image.new("RGB", (64,64), "gray").save(source)
    first = execute(source, dict(session_id="session", number=1))
    assert first.data["image_requests_sent"] == 2
    assert first.data["tool_pipeline_started"] == 1
    parent_record = first.record_path.read_bytes()
    selected = chosen_option(dict(first.data, directory=str(first.path)), "image_1")
    origin = dict(session_id="session", number=2, parent_task_id=first.path.name,
                  parent_candidate_id=selected["id"], initial_task_id=first.path.name)
    second = execute(selected["absolute_path"], origin)
    assert second.path != first.path
    assert second.data["image_requests_sent"] == 2
    assert second.data["tool_pipeline_started"] == 1
    assert second.data["round"]["selected_source_sha256"] == digest(selected["absolute_path"])
    assert second.data["input"]["sha256"] == digest(selected["absolute_path"])
    assert first.record_path.read_bytes() == parent_record
    assert json.loads(second.record_path.read_text(encoding="utf-8"))["round"]["parent_candidate_id"] == "image_1"
    assert "round_started" in second.events_path.read_text(encoding="utf-8")
    # Users may branch from an older round into the next unused round number.
    third = execute(selected["absolute_path"], dict(origin, number=3))
    assert third.data["round"]["number"] == 3
    assert third.data["round"]["initial_task_id"] == first.path.name


@pytest.mark.parametrize("change", [
    dict(session_id="another"),
    dict(parent_candidate_id="missing"),
    dict(initial_task_id="wrong"),
    dict(number=1),
])
def test_invalid_parent_links_fail_before_api(tmp_path, monkeypatch, change):
    execute = prepare_service(tmp_path, monkeypatch, lambda: FakeAPI(accept=True))
    source = tmp_path/"input.png"
    Image.new("RGB", (64,64)).save(source)
    first = execute(source, dict(session_id="session", number=1))
    selected = chosen_option(dict(first.data, directory=str(first.path)))
    origin = dict(session_id="session", number=2, parent_task_id=first.path.name,
                  parent_candidate_id=selected["id"], initial_task_id=first.path.name)
    origin.update(change)
    job = execute(selected["absolute_path"], origin)
    assert job.data["status"] == "execution_failed"
    assert job.data["image_requests_sent"] == job.data["tool_pipeline_started"] == 0
    assert job.record_path.is_file()


def test_original_is_available_when_no_generated_candidate(tmp_path, monkeypatch):
    execute = prepare_service(tmp_path, monkeypatch, lambda: FakeAPI(fail=True))
    source = tmp_path/"input.png"
    Image.new("RGB", (64,64)).save(source)
    first = execute(source)
    assert first.data["status"] == "execution_failed"
    selected = chosen_option(dict(first.data, directory=str(first.path)))
    assert selected["id"] == "original"
    second = execute(selected["absolute_path"],
                     dict(session_id=first.path.name, number=2, parent_task_id=first.path.name,
                          parent_candidate_id="original"))
    assert second.data["round"]["parent_candidate_id"] == "original"
    assert second.data["input"]["sha256"] == digest(selected["absolute_path"])


class RoundAPI(FakeAPI):
    def edit(self, original, prompt, job):
        job.reserve("image")
        path = job.path/f"image_{job.data['image_requests_sent']}.png"
        with Image.open(original) as image:
            level = image.getpixel((0,0))[0] + job.data["image_requests_sent"] * 3
            Image.new("RGB", image.size, (level,)*3).save(path)
        return path

    def compare(self, original, candidate, checks, job):
        accepted = self.accept and job.data["image_requests_sent"] == 2
        return FakeAPI(accept=accepted).compare()


@pytest.mark.parametrize("accepted", [True, False])
def test_workbench_candidate_selection_and_next_round(tmp_path, monkeypatch, accepted):
    from streamlit.testing.v1 import AppTest
    prepare_service(tmp_path, monkeypatch, lambda: RoundAPI(accept=accepted))
    actual_execute, calls = service.execute, []
    def execute(**kwargs):
        calls.append(dict(kwargs))
        return actual_execute(**kwargs)
    monkeypatch.setattr(service, "execute", execute)
    source = tmp_path/"input.png"
    Image.new("RGB", (64,64), "gray").save(source)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py").run(timeout=20)
    app.file_uploader[0].set_value(("input.png",source.read_bytes(),"image/png")).run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    first = app.session_state["snapshot"]
    assert first["status"] == ("completed" if accepted else "needs_review")
    assert len(available_choices(first)) == 2
    default = chosen_option(first)
    assert default["id"] == "image_2"
    assert any(candidate_label(default) in m.value for m in app.markdown)
    others = next(e for e in app.expander if e.label.startswith("其他候选"))
    assert not others.proto.expanded
    assert app.button(key="start_repair").disabled  # Requires a new user instruction.
    old_download = app.get("download_button")[0].proto.url
    key = f"choose:{first['directory']}:image_1"
    app.button(key=key).click().run()
    assert app.session_state["selections"][first["directory"]] == "image_1"
    assert any(candidate_label(chosen_option(first, "image_1")) in m.value for m in app.markdown)
    assert chosen_option(first, "image_1")["status"] == "rejected"
    assert old_download != app.get("download_button")[0].proto.url
    assert len(calls) == 1
    app.text_area[0].set_value("再亮一点，背景保持原样").run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    second = app.session_state["snapshot"]
    selected = chosen_option(first, "image_1")
    assert Path(calls[1]["path"]) == selected["absolute_path"]
    assert calls[1]["requirement"] == "再亮一点，背景保持原样"
    assert second["input"]["sha256"] == digest(selected["absolute_path"])
    assert second["round"]["parent_candidate_id"] == "image_1"
    assert second["round"]["number"] == 2
    assert len(app.session_state["rounds"]) == 2
    app.selectbox(key="view_round").set_value(0).run()
    assert not app.exception
    assert app.session_state["snapshot"]["directory"] == first["directory"]
    assert app.session_state["selections"][first["directory"]] == "image_1"
    app.toggle[0].set_value(True).run()
    assert len(calls) == 2
    app.text_area[0].set_value("从第一轮的所选图片重新修改").run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    third = app.session_state["snapshot"]
    assert third["round"]["number"] == 3
    assert third["round"]["parent_task_id"] == Path(first["directory"]).name
    assert third["round"]["parent_candidate_id"] == "image_1"
    assert third["round"]["initial_task_id"] == Path(first["directory"]).name
    assert Path(calls[2]["path"]) == selected["absolute_path"]


def test_workbench_inherits_and_redraws_region_for_next_round(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ui import region_selector
    prepare_service(tmp_path, monkeypatch, lambda: FakeAPI(accept=True))
    actual_execute, calls, boxes = service.execute, [], []
    def execute(**kwargs):
        calls.append(dict(kwargs))
        return actual_execute(**kwargs)
    def picker(image, image_id, region):
        return dict(image_id=image_id, region=boxes[0] if boxes else region)
    monkeypatch.setattr(service, "execute", execute)
    monkeypatch.setattr(region_selector, "select_region", picker)
    source = tmp_path/"input.png"
    Image.new("RGB", (128,128), "gray").save(source)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py").run(timeout=20)
    app.file_uploader[0].set_value(("input.png",source.read_bytes(),"image/png")).run()
    app.toggle(key="local_mode").set_value(True).run()
    boxes.append(dict(x0=.2, y0=.2, x1=.7, y1=.7))
    app.run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["repair_region"] == boxes[0]
    first = app.session_state["snapshot"]
    boxes[0] = dict(x0=.4, y0=.3, x1=.8, y1=.8)
    app.run()
    assert app.session_state["repair_region"] == boxes[0]
    assert app.session_state["snapshot"] == first
    app.text_area[0].set_value("改善新选区").run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    assert calls[1]["region"] == boxes[0]
    assert calls[1]["origin"]["parent_candidate_id"] == "image_1"
    assert app.session_state["snapshot"]["local_repair"]["selection"] == boxes[0]


def test_workbench_original_fallback_and_new_upload_reset(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    prepare_service(tmp_path, monkeypatch, lambda: FakeAPI(fail=True))
    source = tmp_path/"input.png"
    Image.new("RGB", (64,64)).save(source)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py").run(timeout=20)
    app.file_uploader[0].set_value(("input.png",source.read_bytes(),"image/png")).run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    first = app.session_state["snapshot"]
    assert chosen_option(first)["id"] == "original"
    assert len(app.get("download_button")) == 2
    assert app.button(key="start_repair").label == "基于所选图片继续修改"
    app.text_area[0].set_value("重新去噪").run()
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["snapshot"]["round"]["parent_candidate_id"] == "original"
    assert app.session_state["snapshot"]["image_requests_sent"] == 2
    Image.new("RGB", (80,80), "white").save(source)
    app.file_uploader[0].set_value(("another.png",source.read_bytes(),"image/png")).run()
    assert not app.exception
    assert app.session_state["rounds"] == []
    assert app.session_state["snapshot"] == {}
    assert app.button(key="start_repair").label == "开始修复"
