import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello import refractive_transfer as m


def fixture():
 h,w=30,40;y,x=np.mgrid[:h,:w];plate=np.zeros((h,w,3),np.uint8)+80;target=plate.copy();target[:,:,0]=np.where(x%8<4,110,60);target[:,:,1]=np.where(y%8<4,110,60);support=(x>9)&(x<29)&(y>7)&(y<23);flow=np.zeros((h,w,2),float);flow[support,0]=2;trans=np.ones((h,w,3))*.5;trans[(x>15)&(x<22)&(y>12)&(y<16)]=0;source=m.encode(.08+trans*m.warp(m.decode(plate),flow,support));source[~support]=plate[~support];return source,plate,target,flow,trans,support


def test_identity_background_returns_source_on_support():
 s,p,t,f,a,k=fixture();out,stats=m.transfer_arrays(s,p,p,f,a,k);assert np.array_equal(out[k],s[k]);assert np.array_equal(out[~k],p[~k]);assert stats['zero_transmission_rgb_exact']


def test_refraction_changes_sample_not_opaque_label():
 s,p,t,f,a,k=fixture();out,stats=m.transfer_arrays(s,p,t,f,a,k);zero=k&np.all(a==0,2);assert np.array_equal(out[zero],s[zero]);assert not np.array_equal(out[k],s[k]);assert np.array_equal(out[~k],t[~k]);assert stats['flow_nonzero_pixels']==int(k.sum())


def test_analytic_linear_solution():
 s,p,t,f,a,k=fixture();out,_=m.transfer_arrays(s,p,t,f,a,k);expected=m.encode(m.decode(s)+a*(m.warp(m.decode(t),f,k)-m.warp(m.decode(p),f,k)));assert np.array_equal(out[k],expected[k])

@pytest.mark.parametrize('bad',['flow_nan','flow_shape','flow_outside','trans_nan','trans_range','trans_shape','support_empty','support_nonbool','rgb_shape','rgb_dtype','negative_residual','gamut'])
def test_reject_unsupported_arrays(bad):
 s,p,t,f,a,k=fixture()
 if bad=='flow_nan':f[0,0]=np.nan
 if bad=='flow_shape':f=f[:,:,0]
 if bad=='flow_outside':f[k,0]=100
 if bad=='trans_nan':a[0,0]=np.nan
 if bad=='trans_range':a[0,0]=2
 if bad=='trans_shape':a=a[:,:,0]
 if bad=='support_empty':k[:]=False
 if bad=='support_nonbool':k=k.astype('uint8')
 if bad=='rgb_shape':t=t[:-1]
 if bad=='rgb_dtype':s=s.astype('float32')
 if bad=='negative_residual':s[:]=0;p[:]=255
 if bad=='gamut':s[:]=250;p[:]=0;t[:]=255
 with pytest.raises(ValueError):m.transfer_arrays(s,p,t,f,a,k)


def files(tmp_path):
 s,p,t,f,a,k=fixture();paths={}
 for name,arr in dict(source=s,plate=p,target=t,transmission=np.rint(a*255).astype('uint8'),support=k.astype('uint8')*255).items():
  paths[name]=tmp_path/(name+'.png');Image.fromarray(arr).save(paths[name])
 paths['flow']=tmp_path/'flow.npy';np.save(paths['flow'],f);spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(schema='renderello-refractive-transfer-v1',color_space='srgb',flow_convention='source_pixel_to_background_dxdy_pixels',capture_alignment_reviewed=True,maps_reviewed=True,material='transparent',framing='complete',map_origin='synthetic_known',image_size=[40,30],input_sha256={n:m.sha(p) for n,p in paths.items()})));return paths,spec


def test_files_roundtrip(tmp_path):
 paths,spec=files(tmp_path);out=tmp_path/'out';d=m.transfer_files(**paths,spec=spec,out=out,confirm_assumptions=True);assert d['zero_transmission_rgb_exact'];assert d['real_glass_validated'] is False;assert m.sha(out/'candidate.png')==d['output_sha256']['candidate.png']

@pytest.mark.parametrize('bad',['confirmation','hash','alignment','origin','material','framing','dimensions','flow_convention','existing'])
def test_file_gates(tmp_path,bad):
 paths,spec=files(tmp_path);d=json.loads(spec.read_text());out=tmp_path/'out'
 if bad=='hash':d['input_sha256']['source']='wrong'
 if bad=='alignment':d['capture_alignment_reviewed']=False
 if bad=='origin':d['map_origin']='estimated_without_review'
 if bad=='material':d['material']='unknown'
 if bad=='framing':d['framing']='cropped'
 if bad=='dimensions':d['image_size']=[1,2]
 if bad=='flow_convention':d['flow_convention']='absolute'
 if bad=='existing':out.mkdir()
 spec.write_text(json.dumps(d))
 with pytest.raises(ValueError):m.transfer_files(**paths,spec=spec,out=out,confirm_assumptions=bad!='confirmation')

def test_cli_actual_files(tmp_path,capsys):
 from renderello.cli import main
 paths,spec=files(tmp_path);out=tmp_path/'cli';args=['refractive-transfer','--spec',str(spec),'--out',str(out),'--confirm-assumptions']
 for name,path in paths.items():args+=['--'+name,str(path)]
 assert main(args)==0
 assert json.loads(capsys.readouterr().out)['real_glass_validated'] is False


def test_input_file_change_after_snapshot_cannot_change_decoded_pixels(tmp_path,monkeypatch):
 paths,spec=files(tmp_path);baseline=m.transfer_files(**paths,spec=spec,out=tmp_path/'baseline',confirm_assumptions=True)
 original_open=m.Image.open;changed=False
 def mutate_after_snapshot(file,*args,**kwargs):
  nonlocal changed
  if not changed:
   changed=True
   Image.fromarray(np.zeros((30,40,3),np.uint8)).save(paths['source'])
  return original_open(file,*args,**kwargs)
 monkeypatch.setattr(m.Image,'open',mutate_after_snapshot)
 result=m.transfer_files(**paths,spec=spec,out=tmp_path/'after',confirm_assumptions=True)
 assert changed and baseline['output_sha256']==result['output_sha256']
 assert result['input_sha256']['source']!=m.sha(paths['source'])
 assert result['input_read_semantics']=='bounded_immutable_byte_snapshots_hashed_before_decode'


def test_spec_change_after_snapshot_does_not_relabel_experiment(tmp_path,monkeypatch):
 paths,spec=files(tmp_path);expected=m.sha(spec);real_load=m.np.load
 def mutate(file,*args,**kwargs):
  spec.write_text('{}')
  return real_load(file,*args,**kwargs)
 monkeypatch.setattr(m.np,'load',mutate)
 result=m.transfer_files(**paths,spec=spec,out=tmp_path/'after',confirm_assumptions=True)
 assert result['spec_sha256']==expected and m.sha(spec)!=expected
 assert not result['real_glass_validated']


def test_snapshot_size_bound_applies_before_decode(tmp_path):
 p=tmp_path/'oversized';p.write_bytes(b'12345')
 assert m._snapshot(p,5)==b'12345'
 with pytest.raises(ValueError,match='bounded byte'):m._snapshot(p,4)
