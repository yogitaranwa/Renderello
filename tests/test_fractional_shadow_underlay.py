from dataclasses import replace
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_surface_shadow import supplement_surface_shadow,SurfaceShadow

def setup(alpha_value=190):
 rgb=np.full((60,80,3),120,np.uint8);a=np.zeros((60,80),np.uint8);a[10:40,20:60]=255;a[40,20:60]=alpha_value
 bg=np.full_like(rgb,220);c=np.zeros(a.shape,float);bottom=40 if alpha_value>=128 else 39;c[bottom+1:bottom+4,30:50]=.3
 p=SurfaceShadow(40,40,18,4,.3,digest(rgb),digest(a),digest(bg),digest(c),'Synthetic fractional rim: opacity not a geometry truth.',reviewed=True)
 return rgb,a,bg,c,p

def independent(rgb,a,bg,shade):
 def lin(x):
  x=x/255.;return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 def code(x):return np.rint(np.clip(np.where(x<=.0031308,12.92*x,1.055*np.maximum(0,x)**(1/2.4)-.055),0,1)*255).astype(np.uint8)
 shaded=code(lin(bg)*(1-shade[...,None]));af=a[...,None]/255.;return code(lin(rgb)*af+lin(shaded)*(1-af))

@pytest.mark.parametrize('v',[128,160,190,220,254])
def test_fractional_layer(v):
 rgb,a,bg,c,p=setup(v);old,legacy,_=supplement_surface_shadow(rgb,a,bg,c,p);new,s,m=supplement_surface_shadow(rgb,a,bg,c,p,fractional_underlay=True)
 assert np.array_equal(new,independent(rgb,a,bg,s));assert np.array_equal(old,independent(rgb,a,bg,legacy));assert np.any(s[a==v]>0);assert not np.any(legacy[a==v]);assert m['fractional_underlay'];assert m['fractional_underlay_pixels']>0
 changed=np.any(old!=new,axis=2);assert changed.any();assert not np.any(changed&((a<128)|(a==255)));assert np.array_equal(new[a==255],rgb[a==255]);assert np.all(new.astype(int)<=old.astype(int))

@pytest.mark.parametrize('v',[0,100,127,255])
def test_unaffected_coverage(v):
 args=setup(v);old,_,_=supplement_surface_shadow(*args);new,_,_=supplement_surface_shadow(*args,fractional_underlay=True);assert np.array_equal(old,new)

@pytest.mark.parametrize('v',[1,None,'true',np.bool_(True)])
def test_strict_mode(v):
 with pytest.raises(ValueError):supplement_surface_shadow(*setup(),fractional_underlay=v)

def test_default_exact_and_immutable():
 args=setup();before=[x.copy() for x in args[:4]];old,s,m=supplement_surface_shadow(*args);same,ss,mm=supplement_surface_shadow(*args,fractional_underlay=False);assert np.array_equal(old,same);assert np.array_equal(s,ss);assert m==mm;assert all(np.array_equal(x,y) for x,y in zip(args[:4],before));assert m['shadow_exclusion']=='alpha128_and_above_legacy'

def test_binding_still_enforced():
 args=list(setup());args[4]=replace(args[4],alpha_sha256='0'*64)
 with pytest.raises(ValueError):supplement_surface_shadow(*args,fractional_underlay=True)

def test_geometry_still_enforced():
 args=list(setup());args[4]=replace(args[4],floor_row=50)
 with pytest.raises(ValueError):supplement_surface_shadow(*args,fractional_underlay=True)
