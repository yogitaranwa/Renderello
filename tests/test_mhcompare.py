"""S2 comparison runner: plan mode is zero-spend projection; run mode works
with stub backends (zero network, zero credits)."""
import json
import numpy as np
from PIL import Image

from renderello.mhcompare import plan_compare, run_compare, DEFAULT_COMPARE_SET
from renderello.schemas import Aspect
from renderello.templates import get_template


def _src(tmp_path):
    img = np.full((120, 160, 3), 200, np.uint8)
    img[40:80, 60:100] = (20, 20, 140)
    p = tmp_path / "s.png"
    Image.fromarray(img).save(p)
    return p


def test_plan_compare_zero_spend_projection():
    plan = plan_compare(DEFAULT_COMPARE_SET, plan="creator_monthly")
    assert plan["kind"] == "plan_only_no_spend"
    assert plan["total_credits_listed"] == 5 + 10 + 40
    assert plan["total_projected_usd"] > 0
    tiers = {r["model"]: r["tier_required"] for r in plan["models"]}
    assert tiers["flux-2-klein"] == "free" and tiers["seedream-v4"] == "creator"


def test_plan_compare_rejects_unknown_model():
    try:
        plan_compare(["not-a-model"])
    except KeyError:
        return
    raise AssertionError("unknown model must be rejected before any spend")


def test_run_compare_with_stub_backends(tmp_path):
    src = _src(tmp_path)
    calls = []

    class FakeBackend:
        live = False

        def __init__(self, model):
            self.model = model

        def edit(self, source, prompt, negative_prompt, seed, aspect, reference=None):
            calls.append((self.model, seed, aspect))
            return source.resize(aspect.size)

    out = tmp_path / "cmp"
    manifest = run_compare(src, ["flux-2-klein", "qwen-edit", "seedream-v4"],
                           get_template("clean_studio"), 7, Aspect.SQUARE, out,
                           backend_factory=lambda m: FakeBackend(m))
    assert [c[0] for c in calls] == ["flux-2-klein", "qwen-edit", "seedream-v4"]
    assert all(c[1] == 7 and c[2] == Aspect.SQUARE for c in calls)
    on_disk = json.loads((out / "compare_manifest.json").read_text())
    assert on_disk["kind"] == "mh_model_comparison"
    assert len(on_disk["results"]) == 3
    for r in on_disk["results"]:
        assert r["live"] is False
        assert r["credits_charged"] is None  # mock runs never claim a charge
        assert (out / f"compare_{r['model']}_seed7.png").exists()


def test_failed_compare_persists_id_and_refuses_duplicate(tmp_path):
    import pytest
    class Failing:
        live=True
        last_result={'id':'charged-project','submission':{'credits_charged':10}}
        def edit(self,*args):
            raise TimeoutError('poll timeout')
    out=tmp_path/'cmp'
    with pytest.raises(TimeoutError):
        run_compare(_src(tmp_path),['qwen-edit'],get_template('clean_studio'),0,Aspect.SQUARE,out,
                    backend_factory=lambda m:Failing())
    manifest=json.loads((out/'compare_manifest.json').read_text())
    assert manifest['state']=='interrupted'
    assert manifest['results'][0]['project_id']=='charged-project'
    assert manifest['results'][0]['credits_at_submission']==10
    with pytest.raises(FileExistsError):
        run_compare(_src(tmp_path),['qwen-edit'],get_template('clean_studio'),0,Aspect.SQUARE,out,
                    backend_factory=lambda m:Failing())


def test_cli_live_missing_source_rejected_without_network(tmp_path):
    import pytest
    from renderello.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['mh-compare','--live','--out',str(tmp_path/'out')])
    assert exc.value.code==2
