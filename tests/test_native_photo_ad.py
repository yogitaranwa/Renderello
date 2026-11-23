from pathlib import Path
import numpy as np
import pytest
from renderello.native_photo_ad import AdCopy,render_native_ad,LAYOUTS
from renderello.candidate_diagnostics import digest
from PIL import ImageFont

def available_font():
 for name in ('DejaVuSans.ttf','arial.ttf'):
  try:return ImageFont.truetype(name,18).path
  except OSError:pass
 pytest.skip('No test font file available; install a licensed font to exercise text layout')

FONT=available_font()
PAL=dict(ground='#f6f2ef',surface='#ffffff',ink='#4d3533',secondary='#585254',accent='#73a89a')

def call(rgb=None,copy=None,**kw):
 if rgb is None:rgb=np.full((300,400,3),200,np.uint8)
 args=dict(source_sha256=digest(rgb),layout='square',display_font=FONT,text_font=FONT,bold_font=FONT,palette=PAL)
 args.update(kw)
 return render_native_ad(rgb,copy or AdCopy('Photo study','Carry what\nmatters.','Draft concept, not brand-approved.','Synthetic test photo'),**args)

@pytest.mark.parametrize('layout',list(LAYOUTS))
def test_every_original_pixel_retained(layout):
 rgb=np.random.default_rng(1).integers(0,256,(300,400,3),dtype=np.uint8);original=rgb.copy();out,r=call(rgb,layout=layout);dx,dy=r['integer_translation']
 assert np.array_equal(out[dy:dy+300,dx:dx+400],rgb) and np.array_equal(rgb,original)
 assert r['source_rgb_exact_pixels']==120000 and not r['source_resampled'] and not r['photographic_acceptance']
 assert tuple(r['canvas_size'])==LAYOUTS[layout]['size']

@pytest.mark.parametrize('change',[dict(eyebrow=''),dict(headline='x'*101),dict(headline='one\ntwo\nthree'),dict(disclosure='a\nb'),dict(eyebrow='x\t')])
def test_copy_limits(change):
 args=dict(eyebrow='Photo',headline='Photo',disclosure='Draft',attribution='Synthetic test photo');args.update(change)
 with pytest.raises(ValueError):AdCopy(**args)

@pytest.mark.parametrize('change',[dict(source_sha256='0'*64),dict(layout='unknown'),dict(palette={}),dict(palette=dict(PAL,ink='red'))])
def test_binding_layout_palette(change):
 with pytest.raises(ValueError):call(**change)


def test_source_not_resized_to_fit():
 with pytest.raises(ValueError,match='does not fit'):call(np.zeros((700,900,3),np.uint8))


def test_long_copy_refused_not_truncated():
 with pytest.raises(ValueError,match='headline'):call(copy=AdCopy('Photo','W'*100,'Draft','Synthetic test photo'))


def test_font_snapshot_hashes_reported():
 _,r=call();import hashlib
 assert r['font_sha256']['display']==hashlib.sha256(Path(FONT).read_bytes()).hexdigest()


def test_no_silent_source_conversion():
 with pytest.raises(ValueError):call(np.zeros((40,40,4),np.uint8))
