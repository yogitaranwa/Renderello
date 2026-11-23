from renderello.costs import (route_economics, effective_usd_per_credit,
                              projected_usd_per_output, gross_margin_per_output)


def test_route_economics_mixed():
    rows = [{"seconds": 10, "credits_estimated": 0, "success": True},
            {"seconds": 20, "credits_estimated": 0, "success": False},
            {"seconds": 30, "credits_estimated": 10, "success": True}]
    e = route_economics("scene_compose", rows)
    assert e.runs == 3 and e.successes == 2 and e.retries == 1
    assert e.median_seconds == 20 and e.credits_per_successful_output == 5.0


def test_plan_rates_and_projection():
    assert abs(effective_usd_per_credit("creator_monthly") - 19 / 12000) < 1e-12
    # qwen-edit floor 10 credits at creator monthly included rate
    assert projected_usd_per_output("qwen-edit", "creator_monthly") == round(10 * 19 / 12000, 6)
    # override for a higher-resolution generation
    assert projected_usd_per_output("qwen-edit", "pro_annual", credits_per_generation=20) == round(20 * 25 / 25000, 6)


def test_margin_is_input_driven():
    assert gross_margin_per_output(1.0, 0.25) == 0.75


def test_reference_reaches_editor_backend(tmp_path):
    """Generative reference conditioning plumbing: reference_path must reach the
    editor backend on pure_prompt and hybrid (run itself needs GPU/MH)."""
    import numpy as np
    from PIL import Image
    from renderello.config import RunConfig
    from renderello.pipeline import run_pipeline
    from renderello.schemas import EditRequest, Route
    from renderello.templates import get_template
    img = np.full((300, 400, 3), 220, np.uint8); img[100:200, 150:250] = (30, 30, 160)
    src = tmp_path / "s.png"; Image.fromarray(img).save(src)
    ref = tmp_path / "r.png"; Image.fromarray(np.full((50, 50, 3), 90, np.uint8)).save(ref)
    got = {}
    class Rec:
        name = "rec"
        def edit(self, source, prompt, negative_prompt, seed, aspect, reference=None):
            got["ref"] = reference
            return source.resize(aspect.size)
    cfg = RunConfig(seed=0, variants=1, out_root=tmp_path / "o")
    for route in (Route.PURE_PROMPT, Route.HYBRID):
        got.clear()
        run_pipeline(EditRequest(source_path=src, template=get_template("clean_studio"),
                                 seed=0, reference_path=ref), [route], cfg, editor=Rec())
        assert got.get("ref") is not None and got["ref"].size == (50, 50), route
