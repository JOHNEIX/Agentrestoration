import httpx
import pytest
from PIL import Image
from agentrestoration.core import Job,Settings
from agentrestoration.adapters import Qwen,APIError


def test_edit_http_error_consumes_one_request_and_is_sanitized(tmp_path):
    s=Settings(); job=Job(tmp_path,s)
    image=tmp_path/"in.png"; Image.new("RGB",(16,16)).save(image)
    requests=[]
    def respond(request):
        requests.append(request)
        return httpx.Response(429,json={"message":"secret test token"})
    api=Qwen(s);api.keys={"image":"test-secret","vlm":"test-secret"}
    api.client=httpx.Client(transport=httpx.MockTransport(respond))
    with pytest.raises(APIError,match="HTTP_429"): api.edit(image,"restore",job)
    assert len(requests)==1
    assert job.data["image_requests_sent"]==1
    assert "test-secret" not in (job.path/"record.json").read_text(encoding="utf-8")


def test_missing_key_does_not_consume_budget(tmp_path):
    s=Settings();job=Job(tmp_path,s)
    p=tmp_path/"in.png";Image.new("RGB",(16,16)).save(p)
    api=Qwen(s);api.keys={"image":"","vlm":""}
    with pytest.raises(APIError):api.edit(p,"restore",job)
    assert job.data["image_requests_sent"]==0


def test_provider_error_details_are_saved_without_secrets(tmp_path):
    s=Settings();job=Job(tmp_path,s)
    p=tmp_path/"in.png";Image.new("RGB",(16,16)).save(p)
    api=Qwen(s);api.keys={"image":"test-secret","vlm":"test-key"}
    api.client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(400,json={
        "error":{"code":"InvalidParameter","message":"File size exceeds 10485760 bytes. test-secret https://example.test/image?signature=private data:image/png;base64,YWJjZA=="},
        "request_id":"req-test"})))
    with pytest.raises(APIError,match="InvalidParameter"):
        api.edit(p,"restore",job)
    event=next(e for e in job.data["events"] if e["stage"]=="api_error")
    assert event["provider_code"]=="InvalidParameter" and "10485760" in event["message"]
    text=(job.path/"record.json").read_text(encoding="utf-8")
    assert "test-secret" not in text and "signature=private" not in text and "YWJjZA==" not in text


def test_oversized_input_is_encoded_under_limit(tmp_path):
    import base64
    import random
    randomizer=random.Random(42)
    p=tmp_path/"in.png"
    Image.frombytes("RGB",(128,128),randomizer.randbytes(128*128*3)).save(p)
    original=p.read_bytes()
    s=Settings(image_api_max_bytes=40000);job=Job(tmp_path/"jobs",s)
    payload,metadata=Qwen(s).image_input(p,job,1)
    raw=base64.b64decode(payload.split(",",1)[1])
    assert len(payload.encode())<=s.image_api_max_bytes
    assert metadata["mime"]=="image/jpeg"
    assert len(raw)==metadata["sent_bytes"]
    assert (job.path/metadata["path"]).read_bytes()==raw
    assert p.read_bytes()==original
    assert job.data["image_requests_sent"]==0


def test_visual_verification_request_excludes_detector_claims(tmp_path):
    import json
    from copy import deepcopy
    from agentrestoration.core import Verdict
    p=tmp_path/"in.png";Image.new("RGB",(64,64)).save(p)
    job=Job(tmp_path/"jobs",Settings())
    job.data["requirement"]="Remove noise and keep the scene."
    checks=dict(status="uncertain",reason="unreliable_or_truncated_detection",ratio_error=0,
                original=[dict(**{"class":"elephant"},confidence=.99)],
                candidate=[dict(**{"class":"parking meter"},confidence=.98)],
                matches=[],unmatched_original=[0],unmatched_candidate=[0],
                appearance=dict(status="clear",regions=[]))
    previous=deepcopy(checks)
    requests=[]
    def respond(request):
        body=json.loads(request.content)
        content=body["messages"][0]["content"]
        text=content[0]["text"]
        requests.append(request)
        for claim in ("elephant","parking meter","unreliable_or_truncated_detection",
                      "unmatched_original","confidence"):
            assert claim not in text
        assert '"ratio_error": 0' in text and '"appearance"' in text
        assert job.data["requirement"] in text
        assert "small facial details" in text and "main subjects and scene composition" in text
        assert len(content)==3
        verdict=Verdict(decision="accept",quality_improved=True,structure_preserved=True,
                        appearance_change="normal_repair",issues=[],reason="Scene preserved",retry_instruction="")
        return httpx.Response(200,json={"choices":[{"message":{"content":verdict.model_dump_json()}}]})
    api=Qwen(Settings());api.keys={"image":"fake","vlm":"fake"}
    api.client=httpx.Client(transport=httpx.MockTransport(respond))
    assert api.compare(p,p,checks,job).accepted()
    assert len(requests)==1 and checks==previous
