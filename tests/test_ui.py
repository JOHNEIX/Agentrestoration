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
    assert app.button[0].disabled
    assert len(app.text_area)==1


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
    def execute(path,requirement,task,on_event):
        calls.append(path)
        return run(path,requirement,Settings(),FakeAPI(accept=True),Detection(),Tool(),tmp_path/"jobs",on_event=on_event,
                   log_root=tmp_path/"log/tasks")
    monkeypatch.setattr(service,"execute",execute)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"ui/app.py")
    app.run(timeout=20)
    app.file_uploader[0].set_value(("input.png",source.read_bytes(),"image/png")).run()
    assert not app.button[0].disabled
    app.button[0].click().run(timeout=30)
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
