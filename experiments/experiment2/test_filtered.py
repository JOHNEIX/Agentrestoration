import pytest

from .filtered import select_trials, summarize_metrics, paired_metrics


def rows():
    result = []
    for scene, repeat, psnr, returned in (("one",1,10,False),("one",2,30,False),
                                         ("one",3,99,True),("two",1,40,False)):
        for i, group in enumerate("ABC"):
            result.append(dict(scene=scene, repeat=repeat, group=group,
                               output_is_input=returned if group=="B" else False,
                               output_sha256=f"{scene}-{repeat}-{group}",
                               metrics=dict(psnr=psnr+i, ssim=(psnr+i)/100),
                               lpips=1-(psnr+i)/100))
    return result


def test_filter_keeps_matched_trials_and_original_scene_weighting():
    original = rows()
    kept, excluded = select_trials(original)
    assert len(kept)==9 and len(excluded)==1 and len(original)==12
    assert all((r["scene"],r["repeat"]) != ("one",3) for r in kept)
    stats = summarize_metrics(kept)
    assert all(g["trials"]==3 and g["scenes"]==2 for g in stats.values())
    assert stats["A"]["metrics"]["psnr"]["mean"]==30
    assert stats["A"]["metrics"]["psnr"]["trial_mean"]==pytest.approx(80/3)
    assert stats["B"]["metrics"]["psnr"]["mean"]==31
    paired = paired_metrics(kept)
    assert paired["B-A"]["psnr"]["mean_delta"]==1
    assert paired["C-B"]["lpips"]["mean_delta"]==pytest.approx(-.01)


@pytest.mark.parametrize("case",["duplicate","missing","flag"])
def test_filter_rejects_invalid_pairing_or_missing_flag(case):
    part = rows()
    if case=="duplicate": part.append(part[0])
    if case=="missing": part.pop()
    if case=="flag": part[1].pop("output_is_input")
    with pytest.raises(ValueError):
        select_trials(part)


def test_no_retained_trials_has_no_fabricated_metrics():
    part = rows()
    for row in part:
        if row["group"]=="B": row["output_is_input"]=True
    kept, excluded = select_trials(part)
    assert kept==[] and len(excluded)==4
    stats = summarize_metrics(kept)
    assert all(g["metrics"]["psnr"]["mean"] is None for g in stats.values())
    assert paired_metrics(kept)["B-A"]["lpips"]["ci"] is None
