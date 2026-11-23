from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from renderello.checked_research.withoutbg_matting_diagnostic import run, file_hash, quantize_native_alpha

class Session:
    def get_providers(self):
        return ['CPUExecutionProvider']

class Worker:
    categories = ['flat_scene', 'soft_detail']
    birefnet_categories = {'flat_scene'}
    sessions = {'router': Session(), 'coarse': Session()}
    def _route(self, rgb):
        assert rgb.shape == (5, 7, 3)
        assert rgb.dtype == np.float32
        return 'flat_scene', {}, {}
    def _coarse(self, rgb, feeds, feats):
        return np.array([[[0., 1.], [.5, .25]]], np.float32)

def resize(raw, h, w):
    return np.full((1, h, w), raw[0, 1, 0], np.float32)

@pytest.fixture
def setup(tmp_path):
    source = tmp_path/'source.png'
    rgb = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
    Image.fromarray(rgb).save(source, icc_profile=b'test-profile')
    models = tmp_path/'models'; models.mkdir()
    sidecar = models/'withoutbg-open-weights.onnx.json'
    sidecar.write_text(json.dumps(dict(schema_version=3, pipeline='routed', model_version='test-fixture-not-real')))
    args = dict(source_sha256=file_hash(source), sidecar_sha256=file_hash(sidecar),
                reviewed_diagnostic=True, review_note='Explicit CPU diagnostic, not default.',
                pipeline_factory=lambda *a: Worker(), resize=resize)
    return source, models, tmp_path/'out', args, rgb

def test_run_preserves_rgb_alpha_profile_and_records_override(setup):
    source, models, dest, args, rgb = setup
    r = run(source, models, dest, **args)
    out = Image.open(dest/'component-rgba.png')
    assert np.array_equal(np.array(out)[:, :, :3], rgb)
    assert (np.array(out)[:, :, 3] == 128).all()
    assert out.info['icc_profile'] == b'test-profile'
    assert r['normal_route'] == 'birefnet'
    assert r['executed_route'] == 'FORCED_LEARNED_MATTING_DIAGNOSTIC'
    assert r['photographic_acceptance'] is False
    assert r['automatic_fallback'] is False
    assert r['model_internally_resizes'] is True
    assert r['source_rgb_resampled'] is False

@pytest.mark.parametrize('key,value', [('reviewed_diagnostic', False), ('reviewed_diagnostic', 1), ('review_note', ''), ('review_note', 'a'*2001), ('source_sha256', 'f'*64), ('sidecar_sha256', 'bad')])
def test_requires_review_and_bindings(setup, key, value):
    source, models, dest, args, _ = setup; args[key] = value
    with pytest.raises(ValueError): run(source, models, dest, **args)
    assert not dest.exists()

def test_no_overwrite(setup):
    source, models, dest, args, _ = setup; dest.mkdir(); (dest/'keep').write_text('original')
    with pytest.raises(ValueError, match='preserve'): run(source, models, dest, **args)
    assert (dest/'keep').read_text() == 'original'

@pytest.mark.parametrize('mode', ['L', 'RGBA'])
def test_no_silent_mode_conversion(setup, mode):
    source, models, dest, args, _ = setup
    Image.open(source).convert(mode).save(source); args['source_sha256'] = file_hash(source)
    with pytest.raises(ValueError, match='native RGB'): run(source, models, dest, **args)

def test_pixel_budget(setup):
    source, models, dest, args, _ = setup
    Image.new('RGB', (1001, 1000)).save(source); args['source_sha256'] = file_hash(source)
    with pytest.raises(ValueError, match='1MP'): run(source, models, dest, **args)

@pytest.mark.parametrize('raw', [np.ones((2,2),np.float32), np.ones((1,2,2),np.uint8), np.full((1,2,2), np.nan), np.full((1,2,2),1.01),np.full((1,2,2),-.01)])
def test_invalid_predictions_not_clipped(raw):
    with pytest.raises(ValueError): quantize_native_alpha(raw, 5, 7, resize)

def test_resize_shape_checked():
    with pytest.raises(ValueError):quantize_native_alpha(np.ones((1,2,2),np.float32),5,7,lambda *a: np.ones((2,5,7)))

def test_non_cpu_rejected(setup):
    source, models, dest, args, _ = setup
    class GPU:
        def get_providers(self):return ['CUDAExecutionProvider']
    w = Worker(); w.sessions = {'router': GPU(), 'coarse': Session()}
    args['pipeline_factory'] = lambda *a: w
    with pytest.raises(ValueError, match='CPU'): run(source, models, dest, **args)
    assert not dest.exists()

def test_router_unknown_rejected(setup):
    source, models, dest, args, _ = setup
    class Unknown(Worker):
        def _route(self, rgb):return 'unknown', {}, {}
    args['pipeline_factory'] = lambda *a: Unknown()
    with pytest.raises(ValueError, match='category'):run(source, models, dest, **args)

def test_bad_sidecar_rejected(setup):
    source, models, dest, args, _ = setup
    sidecar=models/'withoutbg-open-weights.onnx.json';sidecar.write_text('{}');args['sidecar_sha256']=file_hash(sidecar)
    with pytest.raises(ValueError, match='schema3'):run(source, models, dest, **args)


def test_cli_requires_deliberate_route_flag_before_run(monkeypatch, capsys):
    from renderello.checked_research import withoutbg_matting_diagnostic as d
    monkeypatch.setattr(d, 'run', lambda *a, **kw: pytest.fail('must not execute before review'))
    with pytest.raises(SystemExit) as e:
        d.main(['--source','x','--model-dir','m','--output-dir','o','--source-sha256','f'*64,
                '--sidecar-sha256','a'*64,'--review-note','Diagnostic only'])
    assert e.value.code == 2
    assert '--reviewed-diagnostic' in capsys.readouterr().err


def test_cli_passes_explicit_native_paths_and_review(monkeypatch, capsys):
    from renderello.checked_research import withoutbg_matting_diagnostic as d
    captured = {}
    def fake(source, models, output, **kw):
        captured.update(source=source, models=models, output=output, **kw)
        return {'default_route_changed': False, 'photographic_acceptance': False}
    monkeypatch.setattr(d,'run',fake)
    result=d.main(['--source','x','--model-dir','m','--output-dir','o','--source-sha256','f'*64,
        '--sidecar-sha256','a'*64,'--review-note','User-reviewed local diagnostic','--reviewed-diagnostic'])
    assert captured['source'] == Path('x')
    assert captured['reviewed_diagnostic'] is True
    assert captured['source_sha256'] == 'f'*64
    assert json.loads(capsys.readouterr().out) == result


def test_structural_flags_persist_with_result_without_alpha_changes(setup):
    source, models, dest, args, rgb = setup
    r = run(source, models, dest, **args)
    assert 'FRAME_TOUCH_NOT_PROOF_OF_COMPLETE_PRODUCT' in r['mask_review']['flags']
    assert r['mask_review']['automatic_acceptance'] is False
    assert r['mask_review']['semantic_completeness'] == 'NOT_EVALUATED'
    assert json.loads((dest/'result.json').read_text())['mask_review'] == r['mask_review']
    assert (np.array(Image.open(dest/'alpha.png')) == 128).all()


def test_small_native_input_retains_explicit_unreviewed_status(setup):
    source, models, dest, args, _ = setup
    Image.new('RGB', (2,2)).save(source)
    args['source_sha256'] = file_hash(source)
    class Tiny(Worker):
        def _route(self, rgb): return 'flat_scene', {}, {}
    args['pipeline_factory'] = lambda *a: Tiny()
    r = run(source, models, dest, **args)
    assert r['mask_review']['flags'] == ['TOO_SMALL_FOR_STRUCTURAL_REVIEW']
    assert r['mask_review']['automatic_acceptance'] is False
    assert Image.open(dest/'component-rgba.png').size == (2,2)
