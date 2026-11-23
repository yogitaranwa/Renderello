import hashlib
import numpy as np
import pytest
from PIL import Image
from renderello.source_locked_rgba_import import load_source_locked_rgba
from renderello.candidate_studio import run, main


@pytest.fixture
def saved(tmp_path):
    rgb = np.full((32, 40, 3), 90, np.uint8)
    alpha = np.zeros((32, 40), np.uint8)
    alpha[8:24, 10:30] = 255
    p = tmp_path/'result.png'
    Image.fromarray(np.dstack([rgb, alpha])).save(p)
    return p, rgb, alpha


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_exact_readback_preserves_pixels_and_returns_copy(saved):
    p, rgb, expected = saved
    alpha, evidence = load_source_locked_rgba(p, rgb, sha(p))
    assert np.array_equal(alpha, expected)
    assert evidence['source_rgb_exact'] and evidence['no_new_inference']
    assert not evidence['automatic_acceptance'] and not evidence['physical_relighting']
    assert evidence['model_provenance'] == 'NOT_AUTHENTICATED_BY_PIXEL_REPLAY'
    alpha[:] = 0
    assert np.array(Image.open(p))[:, :, 3].max() == 255


@pytest.mark.parametrize('where', [(0, 0), (12, 12)])
def test_refuses_repainting_even_under_zero_alpha(saved, where):
    p, rgb, _ = saved
    a = np.array(Image.open(p)); a[where[0], where[1], 0] += 1
    Image.fromarray(a).save(p)
    with pytest.raises(ValueError, match='repaints 1'):
        load_source_locked_rgba(p, rgb, sha(p))


@pytest.mark.parametrize('value', [None, '', 'f'*63, 'F'*64, 'g'*64, 12, True])
def test_requires_encoded_hash(saved, value):
    p, rgb, _ = saved
    with pytest.raises(ValueError, match='SHA-256'):
        load_source_locked_rgba(p, rgb, value)


@pytest.mark.parametrize('kind', ['hash', 'symlink', 'rgb', 'shape', 'icc', 'orientation', 'animated', 'invalid'])
def test_refuses_unsafe_or_transformed_input(saved, tmp_path, kind):
    p, rgb, _ = saved
    expected = sha(p)
    if kind == 'symlink':
        q = tmp_path/'link.png'; q.symlink_to(p); p = q
    elif kind == 'rgb':
        Image.open(p).convert('RGB').save(p)
    elif kind == 'shape':
        Image.new('RGBA', (41, 32)).save(p)
    elif kind == 'icc':
        Image.open(p).save(tmp_path/'new.png', icc_profile=b'not-a-profile'); p = tmp_path/'new.png'
    elif kind == 'orientation':
        exif = Image.Exif(); exif[274] = 6
        Image.open(p).save(tmp_path/'new.png', exif=exif); p = tmp_path/'new.png'
    elif kind == 'animated':
        im = Image.open(p); im.save(tmp_path/'new.png', save_all=True, append_images=[Image.new('RGBA', im.size)], duration=100); p = tmp_path/'new.png'
    elif kind == 'invalid':
        p.write_bytes(b'not an image')
    if kind != 'hash': expected = sha(p)
    else: expected = '0'*64
    with pytest.raises((ValueError, OSError)):
        load_source_locked_rgba(p, rgb, expected)


def test_studio_integration_keeps_review_gates(saved, tmp_path):
    p, rgb, alpha = saved
    s, a = tmp_path/'source.png', tmp_path/'prior.png'
    Image.fromarray(rgb).save(s); Image.fromarray(alpha).save(a)
    result = run(s, tmp_path/'out', {'prior': a}, rgba_results={
        'external': dict(path=p, sha256=sha(p), initial='prior')})
    assert result['accepted_count'] == 0 and result['automatically_selected'] is None
    assert result['source_locked_rgba_imports']['external']['source_rgb_exact']
    assert 'SOURCE_BOUND_REVIEW_SPEC_REQUIRED' in result['candidates']['external']['render_blockers']
    assert (tmp_path/'out/rgba-import-external.json').exists()
    assert (tmp_path/'out/DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()


@pytest.mark.parametrize('item', ['bad', 'a=b=c', '=b=c=d', 'a==c=d', 'a=b==d', 'a=b=c='])
def test_cli_rejects_incomplete_pin(item):
    with pytest.raises(SystemExit) as exc:
        main(['--source', 'x', '--out', 'y', '--rgba-result', item])
    assert exc.value.code == 2


@pytest.mark.parametrize('kind', ['missing_initial', 'bad_request', 'collides', 'repainted'])
def test_studio_rejects_bad_rgba_with_failure_record(saved, tmp_path, kind):
    p, rgb, alpha = saved
    source, prior = tmp_path/'source.png', tmp_path/'prior.png'
    Image.fromarray(rgb).save(source); Image.fromarray(alpha).save(prior)
    request = dict(path=p, sha256=sha(p), initial='prior')
    name = 'external'
    if kind == 'missing_initial': request['initial'] = 'absent'
    elif kind == 'bad_request': request['unexpected'] = True
    elif kind == 'collides': name = 'prior'
    elif kind == 'repainted':
        arr = np.array(Image.open(p)); arr[0, 0, 0] += 1
        Image.fromarray(arr).save(p); request['sha256'] = sha(p)
    with pytest.raises(ValueError):
        run(source, tmp_path/'failed', {'prior': prior}, rgba_results={name: request})
    assert (tmp_path/'failed/INCOMPLETE').exists()
    assert (tmp_path/'failed/failure.json').exists()
    assert not (tmp_path/'failed/DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()


@pytest.mark.parametrize('kind,payload',[(b'gAMA',(100000).to_bytes(4,'big')),(b'sRGB',b'\0'),(b'cHRM',b'\0'*32),(b'cICP',b'\x01\x0d\x00\x01'),(b'mDCV',b'\0'*24),(b'cLLI',b'\0'*8),(b'eXIf',b'II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00')])
def test_interpretation_chunks_refused_even_when_rgb_would_match(saved,kind,payload):
    import zlib
    p,rgb,_=saved;data=p.read_bytes();chunk=len(payload).to_bytes(4,'big')+kind+payload+(zlib.crc32(kind+payload)&0xffffffff).to_bytes(4,'big');p.write_bytes(data[:33]+chunk+data[33:])
    with pytest.raises(ValueError,match='interpretation'):load_source_locked_rgba(p,rgb,sha(p))


def test_high_depth_rgba_refused_before_pillow_truncation(saved):
    import zlib,struct
    p,rgb,_=saved
    def chunk(kind,payload):return len(payload).to_bytes(4,'big')+kind+payload+(zlib.crc32(kind+payload)&0xffffffff).to_bytes(4,'big')
    # Correct 16-bit PNG: high bytes agree with the source, but accepting it
    # would silently truncate original sample precision during Pillow decoding.
    rows=b''.join(b'\0'+b''.join(struct.pack('>HHHH',90*257,90*257,90*257,65535)for x in range(40))for y in range(32))
    p.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',40,32,16,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
    with pytest.raises(ValueError,match='8-bit'):load_source_locked_rgba(p,rgb,sha(p))


@pytest.mark.parametrize('kind',['crc','trailing','missing_end','truncated','duplicate_header'])
def test_invalid_png_container_refused(saved,kind):
    p,rgb,_=saved;data=p.read_bytes()
    if kind=='crc':data=data[:29]+bytes([data[29]^1])+data[30:]
    elif kind=='trailing':data+=b'extra'
    elif kind=='missing_end':data=data[:-12]
    elif kind=='truncated':data=data[:-4]
    else:data=data[:33]+data[8:33]+data[33:]
    p.write_bytes(data)
    with pytest.raises(ValueError):load_source_locked_rgba(p,rgb,sha(p))


def test_opaque_core_result_passes_exact_constraint_without_auto_acceptance(saved,tmp_path):
    import json
    from renderello.candidate_diagnostics import digest
    p,rgb,alpha=saved;source=tmp_path/'source.png';mask=tmp_path/'mask.png';Image.fromarray(rgb).save(source)
    near=alpha.copy();near[alpha==255]=254;Image.fromarray(near).save(mask)
    spec=dict(schema='renderello-candidate-studio-v1',source_pixels_sha256=digest(rgb),candidate_sha256=dict(prior=digest(near),repaired=digest(alpha)),regions=[dict(name='reviewed_opaque_label',box=[14,12,26,20],kind='include',threshold=255,minimum_fraction=1.)],material='opaque',framing='complete',source_pose='unknown',mask_reviewed=False)
    sp=tmp_path/'spec.json';sp.write_text(json.dumps(spec));m=run(source,tmp_path/'studio',{'prior':mask},spec_path=sp,rgba_results={'repaired':dict(path=p,sha256=sha(p),initial='prior')})
    assert not m['candidates']['prior']['constraints'][0]['passed'];assert m['candidates']['repaired']['constraints'][0]['passed']
    assert m['accepted_count']==0 and m['automatically_selected']is None
    assert 'VISUAL_MASK_REVIEW_REQUIRED'in m['candidates']['repaired']['render_blockers']
    assert 'POSE_NOT_ELIGIBLE_FOR_SURFACE_CONTACT'in m['candidates']['repaired']['render_blockers']
