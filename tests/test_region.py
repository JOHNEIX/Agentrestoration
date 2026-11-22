import numpy as np
import pytest
from PIL import Image

from agentrestoration.core import Job, Settings
from agentrestoration.pipeline import run
from agentrestoration.region import RepairRegion, compose_region, prepare_region
from test_core import FakeAPI, Detection, Tool

SELECTION = dict(x0=.25, y0=.25, x1=.75, y1=.75)


def test_normalized_integer_pixels_do_not_expand_from_roundoff():
    selection=RepairRegion(x0=96/321,y0=176/481,x1=224/321,y1=304/481)
    assert selection.pixels((321,481)) == (96,176,224,304)


@pytest.mark.parametrize("selection", [
    dict(x0=.8,y0=0,x1=.2,y1=1), dict(x0=0,y0=.2,x1=1,y1=.2),
    dict(x0=-.1,y0=0,x1=1,y1=1), dict(x0=0,y0=0,x1=1.1,y1=1),
    dict(x0=float("nan"),y0=0,x1=1,y1=1), {},
])
def test_invalid_region(selection):
    with pytest.raises(ValueError):
        RepairRegion.model_validate(selection)


@pytest.mark.parametrize("selection", [SELECTION, dict(x0=0,y0=0,x1=.05,y1=.05),
                                      dict(x0=.95,y0=.95,x1=1,y1=1)])
def test_context_crop_and_exact_outside_pixels(tmp_path, selection):
    pixels = np.random.default_rng(7).integers(0,256,(100,200,3),dtype=np.uint8)
    source = tmp_path/"input.png"
    Image.fromarray(pixels).save(source)
    job = Job(tmp_path/"runs", Settings())
    data = prepare_region(source, selection, job)
    box, crop = data["box"], data["crop_box"]
    assert all(v >= 0 for v in crop)
    assert crop[2] <= 200 and crop[3] <= 100
    raw = job.path/"raw.png"
    Image.new("RGB", ((crop[2]-crop[0])*2,(crop[3]-crop[1])*2),"white").save(raw)
    output, metadata = compose_region(source,raw,data,job,"image_1")
    expected = pixels.copy()
    expected[box[1]:box[3],box[0]:box[2]] = 255
    np.testing.assert_array_equal(np.asarray(Image.open(output)), expected)
    mask = np.asarray(Image.open(job.path/"region_mask.png"))
    assert np.count_nonzero(mask) == (box[2]-box[0])*(box[3]-box[1])
    assert metadata["outside_changed_pixels"] == 0
    assert (job.path/"region.json").is_file()


def test_local_pipeline_validates_composite_and_uses_same_crop_for_retry(tmp_path):
    class API(FakeAPI):
        def __init__(self):
            super().__init__()
            self.inputs = []
            self.compared = []
        def edit(self, original, prompt, job):
            self.inputs.append(original.read_bytes())
            assert "Local restoration" in prompt
            job.reserve("image")
            result=job.path/f"image_{job.data['image_requests_sent']}.png"
            with Image.open(original) as image:
                Image.new("RGB", (image.width*2,image.height*2), "white").save(result)
            return result
        def compare(self, original, candidate, checks, job):
            self.compared.append(candidate)
            assert Image.open(candidate).size == (256,256)
            assert checks["local_repair"]["context_pixels"] == 32
            self.accept = len(self.compared) == 2
            return super().compare(original,candidate,checks,job)
    source = tmp_path/"input.png"
    Image.new("RGB",(256,256),(20,40,60)).save(source)
    api = API()
    job = run(source,"restore",Settings(),api,Detection(),Tool(),tmp_path/"runs",
              region=SELECTION,log_root=tmp_path/"log/tasks")
    assert job.data["status"] == "completed"
    assert len(api.inputs) == 2 and api.inputs[0] == api.inputs[1]
    pixels = np.asarray(Image.open(job.path/"final.png"))
    assert np.all(pixels[64:192,64:192] == 255)
    assert np.all(pixels[:64] == [20,40,60])
    assert job.data["image_requests_sent"] == 2
    assert job.data["tool_pipeline_started"] == 0
    assert all(c["path"].endswith("_composite.png") for c in job.data["candidates"])
    assert all((job.path/c["raw_path"]).exists() for c in job.data["candidates"])


def test_bad_crop_geometry_routes_to_tool_before_pasting(tmp_path):
    class Stretch(FakeAPI):
        def edit(self, original,prompt,job):
            job.reserve("image")
            destination = job.path/"image_1.png"
            Image.new("RGB",(300,100),"white").save(destination)
            return destination
    class LocalTool(Tool):
        name = "Test restoration"
        def run(self,original,job):
            assert Image.open(original).size == (192,192)
            output=job.path/"tool_1.png"
            with Image.open(original) as image:
                Image.new("RGB",image.size,"white").save(output)
            return output
    source=tmp_path/"input.png"; Image.new("RGB",(256,256)).save(source)
    job=run(source,"restore",Settings(),Stretch(accept=True),Detection(),LocalTool(),tmp_path/"runs",
            region=SELECTION,log_root=tmp_path/"log")
    assert job.data["status"] == "completed"
    assert job.data["image_requests_sent"] == job.data["tool_pipeline_started"] == 1
    assert job.data["candidates"][0]["checks"]["scope"] == "region_input"
    assert "path" not in job.data["candidates"][0]
    assert (job.path/"image_1.png").is_file()
    assert (job.path/"tool_1_composite.png").is_file()


def test_invalid_region_stops_before_api(tmp_path):
    class Forbidden:
        def analyze(self,*args): pytest.fail("invalid region must not send requests")
    source=tmp_path/"input.png"; Image.new("RGB",(256,256)).save(source)
    job=run(source,"restore",Settings(),Forbidden(),Detection(),Tool(),tmp_path/"runs",
            region=dict(x0=.8,y0=0,x1=.2,y1=1),log_root=tmp_path/"log")
    assert job.data["status"] == "execution_failed"
    assert job.data["image_requests_sent"] == job.data["tool_pipeline_started"] == 0


def test_vlm_verification_includes_selected_region_closeups(tmp_path,monkeypatch):
    from agentrestoration.adapters import Qwen
    from agentrestoration.core import Verdict
    source=tmp_path/"input.png"; Image.new("RGB",(256,256)).save(source)
    candidate=tmp_path/"candidate.png"; Image.new("RGB",(256,256),"white").save(candidate)
    job=Job(tmp_path/"runs",Settings())
    local=prepare_region(source,SELECTION,job)
    def vlm(paths,instruction,schema,passed_job):
        assert len(paths)==4
        assert Image.open(paths[2]).size == Image.open(paths[3]).size == (128,128)
        assert "selected region" in instruction
        return Verdict(decision="uncertain",quality_improved=None,structure_preserved=None,
                       issues=[],reason="mock",retry_instruction="")
    api=Qwen(Settings())
    monkeypatch.setattr(api,"vlm",vlm)
    assert api.compare(source,candidate,{"local_repair":local},job).decision == "uncertain"


def test_actual_editor_prompt_and_crop_record_match_composite(tmp_path):
    import httpx
    import json
    from io import BytesIO
    from agentrestoration.adapters import Qwen
    class API(Qwen):
        provenance="simulated"
        def analyze(self,*args): return FakeAPI().analyze()
        def compare(self,*args): return FakeAPI(accept=True).compare()
    encoded=BytesIO();Image.new("RGB",(384,384),"white").save(encoded,format="PNG")
    sent=[]
    def respond(request):
        if request.method == "POST":
            sent.append(json.loads(request.content))
            return httpx.Response(200,json={"data":[{"url":"https://example.test/candidate.png"}]})
        return httpx.Response(200,content=encoded.getvalue())
    api=API(Settings());api.keys={"image":"mock-key","vlm":"mock-key"}
    api.client=httpx.Client(transport=httpx.MockTransport(respond))
    source=tmp_path/"input.png";Image.new("RGB",(256,256)).save(source)
    job=run(source,"restore",Settings(),api,Detection(),Tool(),tmp_path/"runs",
            region=SELECTION,log_root=tmp_path/"log")
    assert job.data["status"] == "completed"
    recorded=job.data["edit_requests"][0]
    assert recorded["prompt"] == sent[0]["prompt"] == (job.path/"prompts/image_1.txt").read_text(encoding="utf-8")
    assert recorded["input"]["original_size"] == [192,192]
    assert job.data["candidates"][0]["raw_path"] == recorded["candidate"] == "image_1.png"
    assert job.data["candidates"][0]["path"] == "image_1_composite.png"


def test_region_uses_exif_corrected_coordinates(tmp_path):
    from agentrestoration.core import prepare
    source=tmp_path/"portrait.jpg"
    exif=Image.Exif();exif[274]=6
    Image.new("RGB",(80,120)).save(source,exif=exif)
    job=Job(tmp_path/"runs",Settings())
    normalized=prepare(source,job,Settings())
    local=prepare_region(normalized,dict(x0=0,y0=0,x1=.5,y1=.5),job)
    assert local["box"] == [0,0,60,40]
    assert Image.open(job.path/"region_mask.png").size == (120,80)
