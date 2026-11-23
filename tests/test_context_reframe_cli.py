from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
from renderello import context_reframe_cli as m
from renderello.candidate_diagnostics import digest


def fixture(tmp_path,size=(80,60)):
 w,h=size;rgb=np.full((h,w,3),255,np.uint8);a=np.zeros((h,w),np.uint8);rgb[h//3:2*h//3,w//3:2*w//3]=[10,50,100];a[h//3:2*h//3,w//3:2*w//3]=255
 Image.fromarray(rgb).save(tmp_path/'s.png');Image.fromarray(a).save(tmp_path/'a.png');d=dict(schema='renderello-whole-photo-reframe-review-v1',source_pixels_sha256=digest(rgb),alpha_pixels_sha256=digest(a),material='opaque',reviewed=True,note='Synthetic known array fixture, not a real photo or quality approval.')
 (tmp_path/'r.json').write_text(json.dumps(d));return tmp_path/'s.png',tmp_path/'a.png',tmp_path/'r.json',d,rgb,a


def test_three_exports_whole_frame_exact(tmp_path):
 s,a,r,_,rgb,_=fixture(tmp_path);out=tmp_path/'out';result=m.run(s,a,r,out)
 assert result['rendered_exports']==3 and result['refused_exports']==0 and result['new_template_accepts']==0
 assert not(out/'INCOMPLETE').exists()
 for row in result['rows']:
  image=np.array(Image.open(out/row['file']));dx,dy=row['report']['integer_translation'];h,w=rgb.shape[:2]
  assert np.array_equal(image[dy:dy+h,dx:dx+w],rgb)
  assert row['status']=='EXECUTED_REVIEW_REQUIRED'


@pytest.mark.parametrize('change',[{'reviewed':False},{'reviewed':1},{'material':'transparent'},{'note':''},{'source_pixels_sha256':'a'*64},{'alpha_pixels_sha256':'b'*64},{'unknown':1},{'schema':'wrong'}])
def test_unbound_or_unknown_reviews_refused_without_directory(tmp_path,change):
 s,a,r,d,_,_=fixture(tmp_path);d.update(change);r.write_text(json.dumps(d));out=tmp_path/'out'
 with pytest.raises(ValueError):m.run(s,a,r,out)
 assert not out.exists()


def test_frame_cut_refusals_retained_not_cropped(tmp_path):
 s,a,r,d,rgb,alpha=fixture(tmp_path);alpha[0,0]=255;Image.fromarray(alpha).save(a);d['alpha_pixels_sha256']=digest(alpha);r.write_text(json.dumps(d));out=tmp_path/'out';result=m.run(s,a,r,out)
 assert result['rendered_exports']==0 and result['refused_exports']==3
 assert not list(out.glob('square.png'))
 assert all('border band'in row['reason']for row in result['rows'])


def test_aspect_fit_failures_not_resized(tmp_path):
 s,a,r,_,rgb,_=fixture(tmp_path,size=(1000,800));result=m.run(s,a,r,tmp_path/'out')
 assert result['rendered_exports']==1 and result['refused_exports']==2
 assert result['rows'][0]['aspect']=='square'


@pytest.mark.parametrize('aspects',[(),('square','square'),('other',)])
def test_invalid_aspect_requests(tmp_path,aspects):
 s,a,r,*_=fixture(tmp_path)
 with pytest.raises(ValueError):m.run(s,a,r,tmp_path/'out',aspects=aspects)


def test_existing_export_never_overwritten(tmp_path):
 s,a,r,*_=fixture(tmp_path);out=tmp_path/'out';out.mkdir();(out/'keep').write_text('keep')
 with pytest.raises(FileExistsError):m.run(s,a,r,out)
 assert(out/'keep').read_text()=='keep'


def test_mode_conversion_not_silent(tmp_path):
 s,a,r,*_=fixture(tmp_path);Image.open(s).convert('RGBA').save(s)
 with pytest.raises(ValueError,match='RGB required'):m.run(s,a,r,tmp_path/'out')


def test_cli_manifest_preserves_review_and_snapshot_hashes(tmp_path,capsys):
 s,a,r,*_=fixture(tmp_path);out=tmp_path/'out';original=r.read_bytes();assert m.main(['--source',str(s),'--alpha',str(a),'--review',str(r),'--out',str(out),'--aspects','landscape'])==0
 result=json.loads(capsys.readouterr().out);assert result['planned_exports']==1 and result['rendered_exports']==1
 assert(out/'review.json').read_bytes()==original
 assert result['photo_quality']=='NOT_EVALUATED'and result['distinct_sources']==1
