from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello.reviewed_opaque_core_cli import execute


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def data(tmp_path):
 rgb=np.full((64,80,3),120,np.uint8);a=np.zeros((64,80),np.uint8);a[5:59,5:75]=254;r=np.zeros_like(a);r[20:44,20:60]=255
 paths={}
 for n,v in [('source',rgb),('alpha',a),('region',r)]:
  p=tmp_path/(n+'.png');Image.fromarray(v).save(p);paths[n]=p
 spec=dict(schema='reviewed-opaque-core-files.v1',files={n:sha(p)for n,p in paths.items()},reviewed=True,material='opaque',review_note='Reviewed explicit opaque interior fixture.',controls=dict(max_deficit=8,clearance=8,feather=4));p=tmp_path/'review.json';p.write_text(json.dumps(spec));return dict(**paths,spec=p,spec_sha256=sha(p),output_dir=tmp_path/'result'),spec


def save(args,spec):args['spec'].write_text(json.dumps(spec));args['spec_sha256']=sha(args['spec'])


def test_saved_exact_outputs_and_preserve(data):
 args,s=data;m=execute(**args);p=args['output_dir'];rgb=np.array(Image.open(args['source']));old=np.array(Image.open(args['alpha']));r=np.array(Image.open(args['region']))>0;rgba=np.array(Image.open(p/'component-rgba.png'));a=rgba[:,:,3]
 assert np.array_equal(rgba[:,:,:3],rgb);assert np.array_equal(a[~r],old[~r]);assert a[30,30]==255
 assert m['metrics']['new_opaque_pixels']>0 and not m['accepted']
 for n,h in m['outputs'].items():assert sha(p/n)==h
 for n,bg in [('light',242),('dark',35)]:
  aa=a.astype('uint32')[:,:,None];expected=((rgb.astype('uint32')*aa+bg*(255-aa)+127)//255).astype('uint8');assert np.array_equal(expected,np.array(Image.open(p/(n+'.png'))))
 with pytest.raises(ValueError,match='exists'):execute(**args)


@pytest.mark.parametrize('case',['review','glass','note','hash','extra','missing','deficit','feather','clearance','bool','schema'])
def test_bad_declaration_no_output(data,case):
 args,s=data
 if case=='review':s['reviewed']=False
 elif case=='glass':s['material']='glass'
 elif case=='note':s['review_note']=''
 elif case=='hash':s['files']['alpha']='0'*64
 elif case=='extra':s['extra']=1
 elif case=='missing':del s['files']['region']
 elif case=='deficit':s['controls']['max_deficit']=17
 elif case=='feather':s['controls']['feather']=1
 elif case=='clearance':s['controls']['clearance']=3
 elif case=='bool':s['controls']['max_deficit']=True
 elif case=='schema':s['schema']='accepted'
 save(args,s)
 with pytest.raises(ValueError):execute(**args)
 assert not args['output_dir'].exists()


@pytest.mark.parametrize('case',['fractional','boundary','shape','bad_alpha','changed_source','source_symlink','spec_symlink','spec_hash','duplicate'])
def test_bad_files_no_output(data,case):
 args,s=data
 if case in ['fractional','boundary','shape']:
  a=np.array(Image.open(args['region']))
  if case=='fractional':a[30,30]=100
  elif case=='boundary':a[5,5]=255
  else:a=a[:-1]
  Image.fromarray(a).save(args['region']);s['files']['region']=sha(args['region'])
 elif case=='bad_alpha':
  a=np.array(Image.open(args['alpha']));a[30,30]=100;Image.fromarray(a).save(args['alpha']);s['files']['alpha']=sha(args['alpha'])
 elif case=='changed_source':args['source'].write_bytes(b'changed')
 elif case in ['source_symlink','spec_symlink']:
  n='source'if case=='source_symlink'else'spec';p=args[n].with_name('link-'+args[n].name);p.symlink_to(args[n]);args[n]=p
 save(args,s)
 if case=='spec_hash':args['spec_sha256']='0'*64
 if case=='duplicate':args['spec'].write_text('{"a":1,"a":2}');args['spec_sha256']=sha(args['spec'])
 with pytest.raises((ValueError,OSError)):execute(**args)
 assert not args['output_dir'].exists()
