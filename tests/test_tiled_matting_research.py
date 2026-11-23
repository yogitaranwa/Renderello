from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello import tiled_matting_research as t


def test_starts_cover_source_and_keep_overlap():
 assert t.tile_starts(2325,850,100)==[0,750,1475]
 assert t.tile_starts(850,850,100)==[0]
 assert t.tile_starts(51,30,10)==[0,20,21]

@pytest.mark.parametrize('args',[(0,850,100),(500,2,1),(500,850,0),(500,850,850),(500,True,1)])
def test_bad_layout_rejected(args):
 with pytest.raises(ValueError):t.tile_starts(*args)


def test_identical_overlaps_preserve_native_alpha():
 a=np.arange(120,dtype=np.uint8).reshape(12,10)
 tiles=[(0,0,a[:8]),(0,4,a[4:])]
 stitched,seams=t.stitch_tiles(a.shape,tiles,4,0)
 assert np.array_equal(stitched,a)
 assert seams[0]['max_difference']==0 and seams[0]['box']==[0,4,10,8]


def test_all_intersections_including_diagonal_audited():
 a=np.full((8,8),255,np.uint8)
 tiles=[(0,0,a),(4,0,a),(0,4,a),(4,4,a)]
 stitched,seams=t.stitch_tiles((12,12),tiles,4,0)
 assert len(seams)==6 and np.all(stitched==255)


def test_seam_limit_refuses_mask_not_just_tags_it():
 a=np.full((8,8),100,np.uint8);b=a.copy();b[0,0]=117
 stitched,seams=t.stitch_tiles((12,8),[(0,0,a),(0,4,b)],4,16)
 assert stitched is None and seams[0]['over_limit']==1 and seams[0]['max_difference']==17
 stitched,_=t.stitch_tiles((12,8),[(0,0,a),(0,4,b)],4,17)
 assert stitched is not None

@pytest.mark.parametrize('shape,tiles,overlap,limit',[
 ((20,20),[(0,0,np.zeros((8,8),np.uint8))],4,16),
 ((4,4),[(0,0,np.zeros((8,8),np.uint8))],4,16),
 ((8,8),[(0,0,np.zeros((8,8),np.float32))],4,16),
 ((8,8),[(0,0,np.zeros((8,8),np.uint8))],0,16),
 ((8,8),[(0,0,np.zeros((8,8),np.uint8))],4,256),
 ((8,8),[(True,0,np.zeros((8,8),np.uint8))],4,16),
 ((4000001,1),[(0,0,np.zeros((8,8),np.uint8))],4,16)])
def test_malformed_stitch_refused(shape,tiles,overlap,limit):
 with pytest.raises(ValueError):t.stitch_tiles(shape,tiles,overlap,limit)


def inputs(tmp):
 rgb=np.zeros((30,20,3),np.uint8);rgb[:]=(80,40,180);a=np.zeros((30,20),np.uint8);a[3:27,4:16]=255
 sp,ap=tmp/'rgb.png',tmp/'a.png';Image.fromarray(rgb).save(sp);Image.fromarray(a).save(ap);return sp,ap,rgb,a


def fake_worker(sp,ap,out,**kw):
 assert kw==dict(method='cf',radius=3,edge_colors=False)
 out.mkdir();Image.open(sp).save(out/'source.png');Image.open(ap).save(out/'refined-alpha.png');Image.open(ap).save(out/'trimap.png')


def test_actual_file_route_preserves_source_and_defaults(tmp_path,monkeypatch):
 sp,ap,rgb,a=inputs(tmp_path);monkeypatch.setattr(t,'guarded_file',fake_worker)
 r=t.run(sp,ap,tmp_path/'out',tile_size=20,overlap=5,seam_limit=0)
 assert r['status']=='SEAM_LIMIT_PASSED_NOT_QUALITY_ACCEPTED' and not r['source_rgb_resampled']
 assert r['quality']=='NOT_EVALUATED' and not r['global_solver_equivalence']
 assert np.array_equal(np.array(Image.open(tmp_path/'out/source.png')),rgb)
 assert np.array_equal(np.array(Image.open(tmp_path/'out/stitched-alpha.png')),a)
 with pytest.raises(FileExistsError):t.run(sp,ap,tmp_path/'out')


def test_rejected_actual_route_keeps_tiles_and_no_stitched_file(tmp_path,monkeypatch):
 sp,ap,_,a=inputs(tmp_path)
 def disagree(sp,ap,out,**kw):
  fake_worker(sp,ap,out,**kw)
  if 'tile1-' in str(sp):
   aa=np.array(Image.open(out/'refined-alpha.png'));aa[0,4]=128;Image.fromarray(aa).save(out/'refined-alpha.png');tri=np.array(Image.open(out/'trimap.png'));tri[0,4]=128;Image.fromarray(tri).save(out/'trimap.png')
 monkeypatch.setattr(t,'guarded_file',disagree)
 r=t.run(sp,ap,tmp_path/'out',tile_size=20,overlap=5,seam_limit=0)
 assert r['status']=='REJECTED_SEAM_LIMIT' and not (tmp_path/'out/stitched-alpha.png').exists()
 assert (tmp_path/'out/tile1-cf/refined-alpha.png').exists() and (tmp_path/'out/research-manifest.json').exists()


@pytest.mark.parametrize('kw',[dict(tile_size=1001),dict(tile_size=True),dict(overlap=0),dict(seam_limit=float('nan'))])
def test_invalid_route_does_not_write(tmp_path,kw):
 sp,ap,_,_=inputs(tmp_path)
 with pytest.raises(ValueError):t.run(sp,ap,tmp_path/'out',**kw)
 assert not (tmp_path/'out').exists()


def test_worker_rgb_corruption_rejected_without_publish(tmp_path,monkeypatch):
 sp,ap,_,_=inputs(tmp_path)
 def bad(sp,ap,out,**kw):
  fake_worker(sp,ap,out,**kw);Image.new('RGB',Image.open(sp).size,'red').save(out/'source.png')
 monkeypatch.setattr(t,'guarded_file',bad)
 with pytest.raises(ValueError,match='RGB changed'):t.run(sp,ap,tmp_path/'out',tile_size=20,overlap=5)
 assert not (tmp_path/'out').exists()


def test_source_mode_not_silently_converted(tmp_path):
 sp,ap,_,_=inputs(tmp_path);Image.open(sp).convert('RGBA').save(sp)
 with pytest.raises(ValueError,match='no conversion'):t.run(sp,ap,tmp_path/'out')
 assert not (tmp_path/'out').exists()
