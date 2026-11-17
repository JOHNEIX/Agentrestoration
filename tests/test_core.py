import pytest
from PIL import Image
from agentrestoration.core import Settings, Job, Analysis, Verdict, choose_route, structure
from agentrestoration.pipeline import run


class Detection:
    def __init__(self, a=None, b=None):
        import itertools
        self.values = itertools.cycle([a or [], b or []])
    def detect(self, path):
        return next(self.values)


def box(x=.5, cls="person"):
    return {"class": cls, "confidence": .95, "area": .2, "center": [x, .5]}


@pytest.mark.parametrize("images,tools,severe,available,expected", [
    (1,0,True,True,"tool"), (1,0,False,True,"image"),
    (2,0,False,True,"tool"), (2,1,True,True,"stop"),
    (1,1,True,True,"image"), (2,0,True,False,"stop")])
def test_routing(images, tools, severe, available, expected):
    assert choose_route(severe, available, images, tools) == expected


def test_budget(tmp_path):
    job = Job(tmp_path, Settings())
    job.reserve("image"); job.reserve("image")
    with pytest.raises(RuntimeError): job.reserve("image")
    job.reserve("tool")
    with pytest.raises(RuntimeError): job.reserve("tool")


@pytest.mark.parametrize("size,status", [((200,200),"uncertain"), ((101,100),"fail"), ((120,100),"fail")])
def test_ratio(tmp_path, size, status):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB",(100,100)).save(a); Image.new("RGB",size).save(b)
    assert structure(a,b,Detection(),Settings())["status"] == status


def test_movement_and_uncertainty(tmp_path):
    p = tmp_path / "a.png"; Image.new("RGB",(100,100)).save(p)
    assert structure(p,p,Detection([box(.2)],[box(.4)]),Settings())["severe"]
    assert structure(p,p,Detection([box()],[box(.5,"dog")]),Settings())["status"] == "uncertain"
    assert structure(p,p,Detection([box(.2),box(.7)],[box(.2),box(.7)]),Settings())["status"] == "pass"


class FakeAPI:
    provenance="simulated"
    def __init__(self, fail=False, accept=False): self.fail=fail; self.accept=accept
    def analyze(self, *args):
        return Analysis(degradations=[dict(type="noise",severity="moderate",uncertain=False)], restoration_goal="denoise", preserve=[],edit_prompt="denoise")
    def edit(self, original, prompt, job):
        job.reserve("image")
        if self.fail: raise TimeoutError()
        dest=job.path / f"fake_{job.data['image_requests_sent']}.png"
        Image.open(original).save(dest)
        return dest
    def compare(self,*args):
        return Verdict(decision="accept" if self.accept else "uncertain", quality_improved=self.accept,
                       structure_preserved=self.accept,issues=[],reason="test",retry_instruction="denoise")


class Tool:
    def available(self,*args): return True
    def run(self,*args): raise RuntimeError("tool failed")


@pytest.mark.parametrize("fail,accept,expected", [(True,False,"execution_failed"),(False,False,"needs_review"),(False,True,"completed")])
def test_pipeline(tmp_path, fail, accept, expected):
    p=tmp_path / "in.png"; Image.new("RGB",(100,100)).save(p)
    job=run(p,"denoise",Settings(),FakeAPI(fail,accept),Detection(),Tool(),tmp_path / "runs", log_root=tmp_path/"log/tasks")
    assert job.data["status"] == expected
    assert "group" not in job.data
    assert job.data["image_requests_sent"] <= 2
    assert job.data["tool_pipeline_started"] <= 1
    assert (job.path / "final.png").exists() == accept


def test_invalid_json():
    with pytest.raises(ValueError): Verdict.model_validate_json('{"decision":"accept"}')


def test_one_to_one(tmp_path):
    p=tmp_path / "a.png"; Image.new("RGB",(100,100)).save(p)
    check=structure(p,p,Detection([box(.5),box(.51)],[box(.5)]),Settings())
    assert len(check["matches"])==1
    assert check["status"]=="fail"
    assert check["severe"]
    check=structure(p,p,Detection([box(.5),box(.51)],[box(.5)]),Settings(),baseline_uncertain=True)
    assert check["status"]=="uncertain"


def test_false_accept():
    v=Verdict(decision="accept",quality_improved=True,structure_preserved=None,issues=[],reason="uncertain",retry_instruction="")
    assert not v.accepted()


def test_unsent_failure_does_not_loop(tmp_path):
    class Unsent(FakeAPI):
        calls=0
        def edit(self, original, prompt, job):
            self.calls+=1
            raise ValueError("local parameter invalid")
    p=tmp_path / "a.png"; Image.new("RGB",(100,100)).save(p)
    api=Unsent()
    job=run(p,"test",Settings(),api,Detection(),Tool(),tmp_path/"jobs", log_root=tmp_path/"log/tasks")
    assert api.calls==1
    assert job.data["image_requests_sent"]==0


def test_exif_and_input_preservation(tmp_path):
    from agentrestoration.core import prepare,digest
    p=tmp_path / "a.jpg"
    exif=Image.Exif(); exif[274]=6
    Image.new("RGB",(80,120)).save(p,exif=exif)
    before=digest(p)
    job=Job(tmp_path/"jobs",Settings())
    result=prepare(p,job,Settings())
    assert Image.open(result).size==(120,80)
    assert digest(p)==before


def test_invalid_input(tmp_path):
    from agentrestoration.core import prepare
    p=tmp_path / "a.png"; p.write_text("not an image")
    with pytest.raises(Exception): prepare(p,Job(tmp_path/"jobs",Settings()),Settings())


def test_severe_error_tool_failure_then_remaining_edit(tmp_path):
    sequence=[]
    class StretchFirst(FakeAPI):
        def edit(self, original, prompt, job):
            sequence.append("image")
            job.reserve("image")
            with Image.open(original) as im:
                assert im.size==(100,100)  # every attempt starts from original
                im=im.resize((120,100)) if job.data["image_requests_sent"]==1 else im.copy()
                dest=job.path / f"attempt_{job.data['image_requests_sent']}.png"
                im.save(dest)
            return dest
    class FailingTool(Tool):
        def run(self, original, job):
            sequence.append("tool")
            raise RuntimeError("simulated tool failure")
    p=tmp_path/"in.png";Image.new("RGB",(100,100)).save(p)
    job=run(p,"denoise",Settings(),StretchFirst(accept=True),Detection(),FailingTool(),tmp_path/"jobs", log_root=tmp_path/"log/tasks")
    assert sequence==["image","tool","image"]
    assert job.data["status"]=="completed"
    assert job.data["image_requests_sent"]==2
    assert job.data["tool_pipeline_started"]==1
