import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ui.presentation import flow_html, preview_image


def test_flow_status_and_escaping():
    html=flow_html({"steps":[dict(id="old", title="首次编辑",status="failed",attempt=1,reason="<script>"),
                             dict(id="new",title="工具回退",status="running",attempt=2)]})
    assert 'data-state="failed"' in html and 'data-state="running"' in html
    assert "第 1 次尝试" in html and "第 2 次尝试" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_horizontal_flow_groups_preserve_attempts_and_result_order():
    from xml.etree import ElementTree
    steps = [
        dict(id="prepare", title="图片准备", status="success"),
        dict(id="analysis", title="退化分析", status="success"),
        dict(id="image_1:generate", title="首次编辑", status="success", attempt=1),
        dict(id="image_1:structure", title="结构检查", status="failed", attempt=1, reason="object_position"),
        dict(id="image_1:quality", title="效果验证", status="skipped", attempt=1),
        dict(id="route_1", title="路线选择", status="success", attempt=1),
        dict(id="tool_1:generate", title="工具回退", status="success", attempt=2),
        dict(id="tool_1:structure", title="结构检查", status="success", attempt=2),
        dict(id="tool_1:quality", title="效果验证", status="failed", attempt=2,
             needs_review=True, reason="无法确认 <background>"),
        dict(id="result", title="最终结果", status="failed")]
    root = ElementTree.fromstring("<root>" + flow_html(dict(steps=steps)) + "</root>")
    groups = root.findall(".//section")
    assert [g.attrib["data-attempt"] for g in groups] == ["1", "2"]
    node_ids = [[n.attrib["data-node"] for n in g.iter() if "data-node" in n.attrib] for g in groups]
    assert node_ids == [[s["id"] for s in steps[:6]], [s["id"] for s in steps[6:]]]
    nodes = {n.attrib["data-node"]: n for n in root.iter() if "data-node" in n.attrib}
    assert nodes["image_1:structure"].attrib["data-state"] == "failed"
    assert nodes["image_1:quality"].attrib["data-state"] == "skipped"
    assert nodes["tool_1:generate"].attrib["data-state"] == "success"
    assert nodes["tool_1:quality"].find("summary") is not None
    assert nodes["tool_1:quality"].find("p").text == "无法确认 <background>"
    assert "需复核" in "".join(nodes["tool_1:quality"].itertext())


def test_box_preview_preserves_artifact(tmp_path):
    from PIL import Image
    p=tmp_path/"original.png";Image.new("RGB",(100,100),"white").save(p)
    before=p.read_bytes()
    rendered=preview_image(p,[dict(box=[.1,.1,.8,.8],**{"class":"person"})])
    assert rendered.getpixel((10,10))!=(255,255,255)
    assert p.read_bytes()==before


def test_app_initial_layout():
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py").run(timeout=20)
    assert not app.exception
    assert [h.value for h in app.subheader]==["01  修复设置","02  图片对比","03  处理流程"]
    assert app.button(key="start_repair").disabled
    assert not app.button(key="shutdown_workbench").disabled
    assert len(app.text_area)==1
    assert app.selectbox[0].options == ["自动处理", "合成高斯去噪", "低光增强", "去雾"]
    settings = next(c for c in app.get("column") if [h.value for h in c.subheader] == ["01  修复设置"])
    workspace = next(c for c in app.get("column")
                     if [h.value for h in c.subheader] == ["02  图片对比", "03  处理流程"])
    assert abs(settings.proto.weight - .22) < .01
    assert abs(workspace.proto.weight - .78) < .01


def test_app_result_selection_does_not_rerun_job(tmp_path,monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from PIL import Image
    from test_core import FakeAPI, Detection, Tool
    from agentrestoration.pipeline import run
    from agentrestoration.core import Settings
    from ui import service
    source=tmp_path/"input.png";Image.new("RGB",(64,64)).save(source)
    calls=[]
    def execute(path,requirement,task,on_event,origin=None):
        calls.append(path)
        return run(path,requirement,Settings(),FakeAPI(accept=True),Detection(),Tool(),tmp_path/"jobs",on_event=on_event,
                   log_root=tmp_path/"log/tasks")
    monkeypatch.setattr(service,"execute",execute)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py")
    app.run(timeout=20)
    app.file_uploader[0].set_value(("input.png",source.read_bytes(),"image/png")).run()
    assert not app.button(key="start_repair").disabled
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["snapshot"]["status"]=="completed"
    record_path=Path(app.session_state["snapshot"]["record_file"])
    assert record_path.name=="record.log" and record_path.is_file()
    assert len(app.get("download_button"))==2
    app.toggle[0].set_value(True).run()
    assert len(calls)==1
    assert not app.exception


def test_app_initialization_error_is_displayed(monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from ui import service
    def fail(**kwargs): raise ValueError("configuration error")
    monkeypatch.setattr(service,"execute",fail)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py")
    app.session_state["running"]=True
    app.session_state["pending"]=dict(path="unused",requirement="test",task="auto")
    app.run(timeout=20)
    assert not app.exception
    assert not app.session_state["running"]
    assert app.session_state["snapshot"]["status"]=="execution_failed"


def test_app_config_error_shows_task_log_and_field(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from ui import service
    monkeypatch.setattr(service, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.json").write_text('{"timeout": -1}', encoding="utf-8")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui/app.py")
    app.session_state["running"] = True
    app.session_state["pending"] = dict(path="unused", requirement="restore", task="auto")
    app.run(timeout=20)
    assert not app.exception
    snapshot = app.session_state["snapshot"]
    assert snapshot["status"] == "execution_failed"
    assert Path(snapshot["record_file"]).is_file()
    assert any("timeout" in error.value for error in app.error)
    assert any("任务日志" in caption.value for caption in app.caption)
    assert len(app.get("download_button")) == 1


def test_exit_button_calls_shutdown_once(monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from ui import service
    calls = []
    monkeypatch.setattr(service, "shutdown", lambda: calls.append("shutdown"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui/app.py").run(timeout=20)
    app.button(key="shutdown_workbench").click().run()
    assert not app.exception
    assert calls == ["shutdown"]
    assert app.session_state["exiting"]
    assert any("工作台正在退出" in message.value for message in app.success)
    app.run()
    assert calls == ["shutdown"]
    assert app.button(key="shutdown_workbench").disabled


def test_exit_button_disabled_while_processing():
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "ui/app.py")
    app.session_state["running"] = True
    app.run(timeout=20)
    assert app.button(key="shutdown_workbench").disabled


def test_local_mode_requires_selection_and_passes_region(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from PIL import Image
    from ui import service, region_selector
    from agentrestoration.pipeline import run
    from agentrestoration.core import Settings
    from test_core import FakeAPI, Detection, Tool
    selected = []
    received = []
    def picker(image, image_id, region):
        return dict(image_id=image_id,region=selected[0]) if selected else None
    monkeypatch.setattr(region_selector, "select_region", picker)
    def execute(path, requirement, task, on_event, region=None, origin=None):
        received.append(region)
        return run(path,requirement,Settings(),FakeAPI(accept=True),Detection(),Tool(),tmp_path/"runs",
                   on_event=on_event, region=region,log_root=tmp_path/"log/tasks")
    monkeypatch.setattr(service,"execute",execute)
    source=tmp_path/"input.png";Image.new("RGB",(256,256)).save(source)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py").run(timeout=20)
    app.file_uploader[0].set_value(("input.png",source.read_bytes(),"image/png")).run()
    app.toggle(key="local_mode").set_value(True).run()
    assert app.button(key="start_repair").disabled
    selected.append(dict(x0=.25,y0=.25,x1=.75,y1=.75))
    app.run()
    assert not app.exception
    assert not app.button(key="start_repair").disabled
    app.button(key="start_repair").click().run(timeout=30)
    assert not app.exception
    assert received == selected
    assert app.session_state["snapshot"]["status"]=="completed"
    assert app.session_state["snapshot"]["local_repair"]["box"] == [64,64,192,192]
    # Clearing the next-round selection preserves the completed round.
    selected[0]=None
    app.run()
    assert app.session_state["repair_region"] is None
    assert app.session_state["snapshot"]["status"] == "completed"
    assert app.button(key="start_repair").disabled
