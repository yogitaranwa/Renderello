"""Adversarial intake fixtures. These tests make no photographic-quality claim."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import struct
import zlib
import pytest
from PIL import Image, ImageCms, ImageOps
from renderello.image_intake import inspect_image, save_intake, IntakePolicy, IntakeError


def fixture(tmp_path, mode='RGB', size=(13, 7), fmt='PNG', **kwargs):
    p = tmp_path / ('image.' + fmt.lower())
    image = Image.new(mode, size)
    if mode == 'RGB':
        image.putdata([(i % 256, (i * 3) % 256, (i * 7) % 256) for i in range(size[0] * size[1])])
    image.save(p, format=fmt, **kwargs)
    return p


def test_rgb_exact_and_source_immutable(tmp_path):
    p = fixture(tmp_path); old = p.read_bytes()
    with Image.open(p) as original:
        result = inspect_image(p)
        assert result.image.tobytes() == original.tobytes()
    assert p.read_bytes() == old
    assert result.record['source_sha256'] == hashlib.sha256(old).hexdigest()
    assert not result.record['automatic_publication_pass']


@pytest.mark.parametrize('orientation', range(1, 9))
def test_all_exif_orientations(tmp_path, orientation):
    exif = Image.Exif(); exif[274] = orientation
    p = fixture(tmp_path, fmt='TIFF', exif=exif)
    # TIFF decoder may apply orientation itself; use JPEG for unambiguous reference.
    p = fixture(tmp_path, fmt='JPEG', exif=exif)
    with Image.open(p) as im:
        expected = ImageOps.exif_transpose(im)
        result = inspect_image(p)
        assert result.image.size == expected.size
        assert result.image.tobytes() == expected.tobytes()
        assert 274 not in result.image.getexif()


def test_invalid_orientation(tmp_path):
    exif = Image.Exif(); exif[274] = 9
    with pytest.raises(IntakeError):inspect_image(fixture(tmp_path, fmt='JPEG', exif=exif))


@pytest.mark.parametrize('mode', ['RGBA', 'LA'])
def test_alpha_retained(tmp_path, mode):
    p = fixture(tmp_path, mode=mode)
    with Image.open(p) as im: expected = im.convert('RGBA')
    result = inspect_image(p)
    assert result.image.mode == 'RGBA'
    assert result.image.tobytes() == expected.tobytes()


def test_palette_transparency(tmp_path):
    p = tmp_path / 'palette.png'; im = Image.new('P', (8, 5)); im.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    im.putdata([i % 2 for i in range(40)]); im.save(p, transparency=0)
    result = inspect_image(p)
    with Image.open(p) as im:assert result.image.tobytes() == im.convert('RGBA').tobytes()


def test_grey_expansion_exact(tmp_path):
    p = fixture(tmp_path, mode='L'); result = inspect_image(p)
    assert result.image.mode == 'RGB'
    assert 'lossless_8bit_channel_expansion' in result.record['transformations']


def test_icc_requires_explicit_conversion(tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    p = fixture(tmp_path, icc_profile=profile)
    with pytest.raises(IntakeError):inspect_image(p)
    r = inspect_image(p, IntakePolicy(allow_colour_conversion=True))
    assert r.record['transformations'] == ['icc_to_srgb']
    assert not r.image.info


def test_bad_icc_no_fallback(tmp_path):
    p = fixture(tmp_path, icc_profile=b'not-a-profile')
    with pytest.raises(IntakeError, match='ICC conversion failed'):
        inspect_image(p, IntakePolicy(allow_colour_conversion=True))


def test_cmyk_unprofiled_refused(tmp_path):
    p = fixture(tmp_path, mode='CMYK', fmt='JPEG')
    with pytest.raises(IntakeError):inspect_image(p, IntakePolicy(allow_colour_conversion=True))


@pytest.mark.parametrize('mode', ['I;16', 'I', 'F'])
def test_no_silent_bitdepth_truncation(tmp_path, mode):
    p = fixture(tmp_path, mode=mode, fmt='TIFF')
    with pytest.raises(IntakeError):inspect_image(p)


@pytest.mark.parametrize('payload', [b'', b'hello', b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n'])
def test_corrupt_or_empty(tmp_path, payload):
    p = tmp_path / 'bad'; p.write_bytes(payload)
    with pytest.raises(IntakeError):inspect_image(p)


def test_truncated_jpeg(tmp_path):
    p = fixture(tmp_path, size=(100, 100), fmt='JPEG');p.write_bytes(p.read_bytes()[:-50])
    with pytest.raises(IntakeError):inspect_image(p)


@pytest.mark.parametrize('field,value', [('max_bytes', 10), ('max_pixels', 10), ('max_side', 5), ('max_aspect', 1.1)])
def test_budgets(tmp_path, field, value):
    p = fixture(tmp_path)
    with pytest.raises(IntakeError):inspect_image(p, IntakePolicy(**{field:value}))


def test_animated_webp_refused(tmp_path):
    p = tmp_path / 'motion.webp';a=Image.new('RGB',(10,10),'red');b=Image.new('RGB',(10,10),'blue')
    a.save(p, save_all=True, append_images=[b], duration=100, loop=0)
    with pytest.raises(IntakeError):inspect_image(p)


def test_multipage_tiff_refused(tmp_path):
    p=tmp_path/'stack.tif';a=Image.new('RGB',(10,10));a.save(p, save_all=True, append_images=[a.copy()])
    with pytest.raises(IntakeError):inspect_image(p)


def test_metadata_not_exported(tmp_path):
    exif=Image.Exif();exif[270]='private description';p=fixture(tmp_path,fmt='JPEG',exif=exif)
    result=inspect_image(p);dest=save_intake(result,tmp_path/'output')
    with Image.open(dest/'normalized.png') as im:
        assert not im.getexif()
    assert 'private description' not in (dest/'intake.json').read_text()


def test_no_overwrite(tmp_path):
    result=inspect_image(fixture(tmp_path));dest=save_intake(result,tmp_path/'out');old=(dest/'normalized.png').read_bytes()
    with pytest.raises(FileExistsError):save_intake(result,dest)
    assert (dest/'normalized.png').read_bytes()==old


def test_concurrent_output_collision(tmp_path):
    p=fixture(tmp_path);dest=tmp_path/'collision'
    def run(_):
        try:save_intake(inspect_image(p),dest);return 'ok'
        except FileExistsError:return 'exists'
    with ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(run,range(8)))
    assert out.count('ok')==1
    assert not (dest/'INTAKE_INCOMPLETE').exists()
    with Image.open(dest/'normalized.png') as im:im.load()


@pytest.mark.parametrize('kwargs', [{'max_bytes':0},{'max_pixels':-1},{'max_side':True},{'max_aspect':float('nan')},{'formats':()}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):IntakePolicy(**kwargs)


def test_rgba_icc_alpha_unchanged(tmp_path):
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    p=fixture(tmp_path,mode='RGBA',icc_profile=profile)
    result=inspect_image(p,IntakePolicy(allow_colour_conversion=True))
    with Image.open(p) as im:assert result.image.getchannel('A').tobytes()==im.getchannel('A').tobytes()


def test_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):inspect_image(tmp_path/'missing')


def test_extension_does_not_determine_type(tmp_path):
    p=fixture(tmp_path);q=tmp_path/'not-an-image.txt';q.write_bytes(p.read_bytes())
    assert inspect_image(q).record['input_format']=='PNG'


def test_rgb16_png_cannot_hide_behind_rgb_mode(tmp_path):
    # A real valid RGB 16-bit PNG, not only an I;16 greyscale image.
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,16,2,0,0,0))+chunk(b'IDAT',zlib.compress(b'\0\x12\x34\x56\x78\x9a\xbc'))+chunk(b'IEND',b'')
    p=tmp_path/'rgb16.png';p.write_bytes(data)
    with Image.open(p) as im:assert im.mode=='RGB'
    with pytest.raises(IntakeError,match='High-bit-depth PNG'):inspect_image(p)


@pytest.mark.parametrize('colour', [(0,0,0),(255,255,255),(255,0,255)])
def test_uniform_black_white_magenta_exact(tmp_path,colour):
    p=tmp_path/'uniform.png';Image.new('RGB',(19,23),colour).save(p)
    r=inspect_image(p);assert r.image.tobytes()==bytes(colour)*437


def test_cli_real_round_trip(tmp_path):
    from renderello.image_intake import main
    p=fixture(tmp_path);old=p.read_bytes();main(['--source',str(p),'--out',str(tmp_path/'cli')])
    assert p.read_bytes()==old
    assert (tmp_path/'cli/intake.json').is_file()


def test_fifo_rejected_before_blocking(tmp_path):
    import os
    if not hasattr(os, 'mkfifo'):
        pytest.skip('FIFO not available on this platform')
    pipe = tmp_path / 'fifo.png'
    os.mkfifo(pipe)
    with pytest.raises(IntakeError, match='regular file'):
        inspect_image(pipe)
