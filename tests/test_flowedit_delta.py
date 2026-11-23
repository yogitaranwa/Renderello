import numpy as np
import pytest
from renderello.flowedit_delta import integrate


def run(x=None,sigmas=(1.,.7,.2,0.),n_max=3,n_avg=1,velocity=None,noise=None,events=None):
 if x is None:x=np.array([1.,2.,3.],dtype=np.float32)
 def check(a):
  if not isinstance(a,np.ndarray) or a.shape!=x.shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('bad array')
 return integrate(x,sigmas,velocity or (lambda a,s,b,i:a*0),noise or (lambda a,i,d:a*0+7),n_max=n_max,n_avg=n_avg,validate_array=check,on_step=(lambda i,s,a:events.append((i,s,a.copy())))if events is not None else None)


def test_null_edit_exact_identity_and_source_unchanged():
 x=np.array([.1134,-21.3,1234.2],dtype=np.float32);old=x.copy();out=run(x=x,velocity=lambda a,s,b,i:a*.3+s)
 assert np.array_equal(out,x) and np.array_equal(x,old) and out is not x


def test_constant_field_sign_and_full_transport():
 out=run(velocity=lambda a,s,b,i:a*0+(2 if b=='target'else 0));assert np.allclose(out,[-1,0,1],atol=1e-6)


def test_partial_steps_use_sigma_not_uniform_timestep():
 events=[];out=run(n_max=2,velocity=lambda a,s,b,i:a*0+(1 if b=='target'else 0),events=events);assert np.allclose(out,[.3,1.3,2.3]);assert [e[0]for e in events]==[1,2]


def test_same_noise_and_parallelogram_state():
 seen=[]
 def v(a,s,b,i):seen.append((b,a.copy()));return a*0+(1 if b=='target'else 0)
 run(velocity=v)
 assert np.array_equal(seen[0][1],seen[1][1]);assert np.allclose(seen[3][1]-seen[2][1],-.3)


def test_n_avg_draws_and_velocity_calls():
 calls=[];noise=[]
 def v(a,s,b,i):calls.append((i,b));return a*0
 def n(a,i,d):noise.append((i,d));return a*0+d
 run(n_avg=3,velocity=v,noise=n);assert len(calls)==18 and len(noise)==9 and len(set(noise))==9


def test_zero_steps_no_callbacks():
 def fail(*a):raise AssertionError('unexpected')
 assert np.array_equal(run(n_max=0,velocity=fail,noise=fail),[1,2,3])

@pytest.mark.parametrize('sigmas',[(),(0.,),(-1.,0.),(1.1,0.),(1.,.5,.5,0.),(1.,.6,.7,0.),(1.,.1),(True,0.),(float('nan'),0.),(float('inf'),0.),('1',0.)])
def test_invalid_schedule(sigmas):
 with pytest.raises(ValueError):run(sigmas=sigmas)

@pytest.mark.parametrize('n',[-1,4,True,2.0,None])
def test_invalid_nmax(n):
 with pytest.raises(ValueError):run(n_max=n)

@pytest.mark.parametrize('n',[0,17,True,2.0,None])
def test_invalid_navg(n):
 with pytest.raises(ValueError):run(n_avg=n)

@pytest.mark.parametrize('bad',[np.array([np.nan]*3,dtype=np.float32),np.ones(4,dtype=np.float32),np.ones(3,dtype=np.float64),[1,2,3]])
def test_bad_velocity_arrays(bad):
 with pytest.raises(ValueError):run(velocity=lambda *args:bad)


def test_nonfinite_noise_refused_before_velocity():
 def fail(*args):raise AssertionError('model called')
 with pytest.raises(ValueError):run(noise=lambda a,i,d:a*float('nan'),velocity=fail)


def test_euler_affine_matches_manual_oracle():
 x=np.array([1.,2.,3.],dtype=np.float32);state=x.copy();schedule=[1.,.7,.2,0.]
 for i,s in enumerate(schedule[:-1]):
  noise=x*0+7;src=(1-s)*x+s*noise;tar=src+(state-x);delta=(2*tar+3)-(src+1);state=state+(schedule[i+1]-s)*delta
 out=run(x=x,velocity=lambda a,s,b,i:2*a+3 if b=='target'else a+1)
 assert np.allclose(out,state,rtol=1e-6,atol=1e-6)
