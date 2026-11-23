import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,PngImagePlugin
import pytest
from renderello.reviewed_edge_cli import execute,load_spec,read_image,sha

@pytest.fixture
def case(tmp_path):
 rgb=np.full((50,60,3),240,np.uint8);a=np.zeros((50,60),np.uint8);a[8:42,10:50]=255;rgb[a==255]=[80,85,75]
 t=np.zeros(a.shape,bool);t[16:34,10:12]=True;rgb[t]=[30,35,25];d=np.zeros(a.shape,bool);d[16:34,15:17]=True;p=np.zeros(a.shape,bool);p[15:35,30:40]=True
 paths={}
 for n,v in zip(('source','alpha','target','donor','protected'),(rgb,a,t,d,p)):
  path=tmp_path/(n+'.png');Image.fromarray(v.astype(np.uint8)*255 if v.dtype==bool else v).save(path);paths[n]=path
 s=dict(schema='reviewed-edge-files.v1',operation='lift',files={n:sha(p.read_bytes()) for n,p in paths.items()},reviewed=True,material='opaque',review_note='Synthetic review, not physical truth.',controls=dict(edge_width=2,donor_distance=24,strength=.5,maximum_gain=1.25,max_code_delta=24))
 sp=tmp_path/'spec.json';sp.write_text(json.dumps(s));kw=dict(**paths,spec=sp,spec_sha256=sha(sp.read_bytes()),output_dir=tmp_path/'out')
 return kw,s

def update(kw,s):kw['spec'].write_text(json.dumps(s));kw['spec_sha256']=sha(kw['spec'].read_bytes())

def test_lift_full_file_replay(case):
 kw,s=case;m=execute(**kw);D=kw['output_dir'];assert len(list(D.iterdir()))==4
 out=np.array(Image.open(D/'foreground.png'));rgb=np.array(Image.open(kw['source']));t=np.array(Image.open(kw['target']))==255;p=np.array(Image.open(kw['protected']))==255
 assert np.array_equal(out[~t],rgb[~t]) and np.array_equal(out[p],rgb[p]) and np.all(out[t]>=rgb[t])
 assert np.array_equal(np.array(Image.open(D/'alpha.png')),np.array(Image.open(kw['alpha'])))
 assert not m['accepted'] and m['review_is_caller_declaration']
 assert np.array_equal(np.array(Image.open(D/'component-rgba.png'))[:,:,:3],out)
 for n,h in m['outputs'].items():assert sha((D/n).read_bytes())==h
 kw['output_dir']=D.with_name('other');assert execute(**kw)==m

def test_darken_full_file_replay(case):
 kw,s=case;s['operation']='darken';s['controls'].pop('maximum_gain');s['controls']['minimum_gain']=.65;update(kw,s)
 m=execute(**kw);assert m['operation']=='darken' and m['metrics']['darken_only']

@pytest.mark.parametrize('name',['source','alpha','target','donor','protected','spec'])
def test_changed_file_rejected_before_reservation(case,name):
 kw,s=case;kw[name].write_bytes(kw[name].read_bytes()+b'x')
 with pytest.raises(ValueError,match='hash'):execute(**kw)
 assert not kw['output_dir'].exists()

@pytest.mark.parametrize('kind',['extra','missing','operation','schema','review','material','control','files','badsha','controltype','root'])
def test_bad_spec(case,kind):
 kw,s=case
 if kind=='extra':s['extra']=1
 if kind=='missing':s.pop('review_note')
 if kind=='operation':s['operation']='auto'
 if kind=='schema':s['schema']='other'
 if kind=='review':s['reviewed']=1
 if kind=='material':s['material']='glass'
 if kind=='control':s['controls']['unknown']=1
 if kind=='files':s['files'].pop('alpha')
 if kind=='badsha':s['files']['alpha']='X'*64
 if kind=='controltype':s['controls']=[]
 if kind=='root':s=[]
 update(kw,s)
 with pytest.raises(ValueError):execute(**kw)
 assert not kw['output_dir'].exists()

@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','[','null'])
def test_duplicate_nonfinite_invalid_json(case,text):
 kw,s=case;kw['spec'].write_text(text);kw['spec_sha256']=sha(kw['spec'].read_bytes())
 with pytest.raises(ValueError):execute(**kw)
 assert not kw['output_dir'].exists()

@pytest.mark.parametrize('kind',['softmask','palette','rgba','sixteen','orientation','gamma','animated','large','shortaxis','jpeg','shape'])
def test_bad_images(case,kind):
 kw,s=case;n='target';p=kw[n]
 if kind=='softmask':Image.fromarray(np.full((50,60),128,np.uint8)).save(p)
 if kind=='palette':Image.new('P',(60,50)).save(p)
 if kind=='rgba':n='source';p=kw[n];Image.new('RGBA',(60,50)).save(p)
 if kind=='sixteen':Image.fromarray(np.zeros((50,60),np.uint16)).save(p)
 if kind=='orientation':im=Image.new('L',(60,50));ex=Image.Exif();ex[274]=6;im.save(p,exif=ex)
 if kind=='gamma':im=Image.new('L',(60,50));info=PngImagePlugin.PngInfo();info.add(b'gAMA',(45455).to_bytes(4,'big'));im.save(p,pnginfo=info)
 if kind=='animated':Image.new('L',(60,50)).save(p,save_all=True,append_images=[Image.new('L',(60,50),255)])
 if kind=='large':Image.new('L',(1001,1001)).save(p)
 if kind=='shortaxis':Image.new('L',(1,50)).save(p)
 if kind=='jpeg':Image.new('L',(60,50)).save(p,format='JPEG')
 if kind=='shape':Image.new('L',(60,49)).save(p)
 s['files'][n]=sha(p.read_bytes());update(kw,s)
 with pytest.raises(ValueError):execute(**kw)
 assert not kw['output_dir'].exists()

def test_existing_preserved(case):
 kw,s=case;D=kw['output_dir'];D.mkdir();(D/'keep').write_text('keep')
 with pytest.raises(ValueError):execute(**kw)
 assert (D/'keep').read_text()=='keep'

def test_symlink_refused(case,tmp_path):
 kw,s=case;kw['output_dir'].symlink_to(tmp_path/'missing')
 with pytest.raises(ValueError):execute(**kw)
 assert not (tmp_path/'missing').exists()

def test_cli_subprocess(case):
 kw,s=case;args=[sys.executable,'-m','renderello.reviewed_edge_cli']
 for k,v in kw.items():args.extend(['--'+k.replace('_','-'),str(v)])
 p=subprocess.run(args,capture_output=True,text=True);assert p.returncode==0,p.stderr
 assert json.loads(p.stdout)['operation']=='lift'
 p=subprocess.run(args,capture_output=True,text=True);assert p.returncode==2 and 'REFUSED' in p.stderr

def test_oversize_spec(case):
 kw,s=case;kw['spec'].write_bytes(b' '*65537);kw['spec_sha256']=sha(kw['spec'].read_bytes())
 with pytest.raises(ValueError):execute(**kw)
 assert not kw['output_dir'].exists()

def test_validation_does_not_create_parent(case):
 kw,s=case;kw['output_dir']=kw['output_dir'].parent/'missing'/'out';s['material']='glass';update(kw,s)
 with pytest.raises(ValueError):execute(**kw)
 assert not kw['output_dir'].parent.exists()

def test_staging_cleanup_on_write_failure(case,monkeypatch):
 kw,s=case
 def fail(*args,**kwargs):raise OSError('simulated write failure')
 monkeypatch.setattr(Image.Image,'save',fail)
 with pytest.raises(OSError):execute(**kw)
 assert not kw['output_dir'].exists() and not list(kw['output_dir'].parent.glob('.edge-stage-*'))

def test_cleanup_only_own_reserved_output(case,monkeypatch):
 import renderello.reviewed_edge_cli as cli
 kw,s=case;real=cli.os.rename;calls=[]
 def fail(src,dest):
  calls.append(src)
  if len(calls)==2:raise OSError('simulated rename failure')
  return real(src,dest)
 monkeypatch.setattr(cli.os,'rename',fail)
 with pytest.raises(OSError):execute(**kw)
 assert not kw['output_dir'].exists() and not list(kw['output_dir'].parent.glob('.edge-stage-*'))

def test_competing_output_not_deleted(case,monkeypatch):
 kw,s=case;D=kw['output_dir'];real=Path.mkdir
 def mkdir(path,*args,**kwargs):
  if path==D:
   real(D);(D/'competitor').write_text('keep');raise FileExistsError('competitor')
  return real(path,*args,**kwargs)
 monkeypatch.setattr(Path,'mkdir',mkdir)
 with pytest.raises(FileExistsError):execute(**kw)
 assert (D/'competitor').read_text()=='keep' and not list(D.parent.glob('.edge-stage-*'))
