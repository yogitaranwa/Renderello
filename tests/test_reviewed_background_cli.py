import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from renderello.reviewed_background_cli import execute, load_spec


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    rgb = np.full((64, 80, 3), 120, np.uint8)
    alpha = np.zeros((64, 80), np.uint8); alpha[20:40, 25:45] = 255; alpha[20:30, 70:78] = 5
    bg = np.zeros_like(alpha); bg[:, 65:] = 255
    keep = np.zeros_like(alpha); keep[20:40, 25:45] = 255
    paths = {}
    for n, a in [('source', rgb), ('alpha', alpha), ('background', bg), ('protected', keep)]:
        p = tmp_path/(n+'.png'); Image.fromarray(a).save(p); paths[n] = p
    spec = dict(schema='reviewed-background-files.v1', files={n: sha(p) for n, p in paths.items()},
        reviewed=True, material='opaque', review_note='Explicit far-right background stripe reviewed for this fixture.',
        controls=dict(ceiling=32, clearance=12, seam_ceiling=2))
    p = tmp_path/'spec.json'; p.write_text(json.dumps(spec))
    return dict(**paths, spec=p, spec_sha256=sha(p), output_dir=tmp_path/'out'), spec, rgb, alpha


def write_spec(args, s):
    args['spec'].write_text(json.dumps(s)); args['spec_sha256'] = sha(args['spec'])


def test_exact_scope_and_outputs(setup):
    args, s, rgb, old = setup
    m = execute(**args); root = args['output_dir']
    a = np.array(Image.open(root/'alpha.png')); rgba = np.array(Image.open(root/'component-rgba.png'))
    assert np.array_equal(rgba[:, :, :3], rgb)
    assert np.array_equal(a[:, :65], old[:, :65]) and not a[:, 65:].any()
    assert m['metrics']['alpha_changed_pixels'] == 80 and m['source_rgb_exact']
    assert not m['accepted'] and not m['default_pipeline_changed']
    for n, h in m['outputs'].items(): assert sha(root/n) == h
    for n, b in [('dark', 35), ('light', 242)]:
        aa = a.astype('uint32')[:, :, None]
        expected = ((rgb.astype('uint32')*aa+b*(255-aa)+127)//255).astype('uint8')
        assert np.array_equal(np.array(Image.open(root/(n+'.png'))), expected)
    with pytest.raises(ValueError, match='exists'): execute(**args)


@pytest.mark.parametrize('kind', ['unreviewed', 'glass', 'note', 'ceiling', 'clearance', 'seam_control', 'extra', 'missing', 'hash', 'bad_schema', 'float_control', 'bool_control'])
def test_refuses_bad_declarations_before_output(setup, kind):
    args, s, _, _ = setup
    if kind == 'unreviewed': s['reviewed'] = False
    elif kind == 'glass': s['material'] = 'glass'
    elif kind == 'note': s['review_note'] = ''
    elif kind == 'ceiling': s['controls']['ceiling'] = 33
    elif kind == 'clearance': s['controls']['clearance'] = 3
    elif kind == 'seam_control': s['controls']['seam_ceiling'] = 5
    elif kind == 'extra': s['unexpected'] = True
    elif kind == 'missing': del s['files']['protected']
    elif kind == 'hash': s['files']['alpha'] = '0'*64
    elif kind == 'bad_schema': s['schema'] = 'accepted'
    elif kind == 'float_control': s['controls']['ceiling'] = 32.0
    elif kind == 'bool_control': s['controls']['ceiling'] = True
    write_spec(args, s)
    with pytest.raises(ValueError): execute(**args)
    assert not args['output_dir'].exists()


@pytest.mark.parametrize('kind', ['overlap', 'high_alpha', 'seam', 'fractional', 'shape', 'repaint_file', 'symlink', 'spec_hash', 'duplicate_spec'])
def test_refuses_invalid_files_and_regions(setup, kind):
    args, s, _, _ = setup
    if kind in ('overlap', 'fractional', 'shape'):
        a = np.array(Image.open(args['background']))
        if kind == 'overlap': a[25, 30] = 255
        elif kind == 'fractional': a[0, 0] = 1
        else: a = a[:60]
        Image.fromarray(a).save(args['background']); s['files']['background'] = sha(args['background'])
    elif kind in ('high_alpha', 'seam'):
        a = np.array(Image.open(args['alpha'])); a[25, 72 if kind == 'high_alpha' else 65] = 40 if kind == 'high_alpha' else 10
        Image.fromarray(a).save(args['alpha']); s['files']['alpha'] = sha(args['alpha'])
    elif kind == 'repaint_file': args['source'].write_bytes(b'changed')
    elif kind == 'symlink':
        q = args['source'].with_name('link.png'); q.symlink_to(args['source']); args['source'] = q
    write_spec(args, s)
    if kind == 'spec_hash': args['spec_sha256'] = '0'*64
    elif kind == 'duplicate_spec':
        args['spec'].write_text('{"schema":1,"schema":2}'); args['spec_sha256'] = sha(args['spec'])
    with pytest.raises((ValueError, OSError)): execute(**args)
    assert not args['output_dir'].exists()
