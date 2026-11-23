import ctypes
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from renderello import process_memory as pm
from renderello import matting as m


class Fn:
    def __init__(self, fn): self.fn=fn;self.calls=[]
    def __call__(self,*args):self.calls.append(args);return self.fn(*args)


def api(*,opened=123,rss=10000,sample_ok=True,close_ok=True):
    def sample(handle,ptr,size):
        assert handle==123 and size==ctypes.sizeof(pm.ProcessMemoryCounters)
        ctypes.cast(ptr,ctypes.POINTER(pm.ProcessMemoryCounters)).contents.WorkingSetSize=rss
        return sample_ok
    k=SimpleNamespace(OpenProcess=Fn(lambda *args:opened),CloseHandle=Fn(lambda *args:close_ok))
    p=SimpleNamespace(GetProcessMemoryInfo=Fn(sample))
    return k,p


def test_windows_layout_and_resident_not_peak():
    assert ctypes.sizeof(pm.ProcessMemoryCounters)==8+8*ctypes.sizeof(ctypes.c_size_t)
    assert pm.ProcessMemoryCounters.WorkingSetSize.offset==8+ctypes.sizeof(ctypes.c_size_t)
    k,p=api(rss=2**33+17)
    r=pm.ResidentMemory(99,platform='win32',windows_api=lambda:(k,p))
    assert r.sample()==2**33+17
    assert k.OpenProcess.calls==[(0x1000,False,99)]
    r.close();r.close();assert len(k.CloseHandle.calls)==1
    with pytest.raises(RuntimeError,match='closed'):r.sample()


def test_windows_ctypes_signatures(monkeypatch):
    k,p=api();monkeypatch.setattr(ctypes,'WinDLL',lambda name,**kw:k if name=='kernel32' else p,raising=False)
    assert pm._windows_api()==(k,p)
    assert k.OpenProcess.restype==ctypes.c_void_p
    assert p.GetProcessMemoryInfo.argtypes[2]==ctypes.c_uint32
    assert k.CloseHandle.argtypes==[ctypes.c_void_p]


def test_windows_open_refused():
    k,p=api(opened=0)
    with pytest.raises(RuntimeError,match='handle unavailable'):pm.ResidentMemory(7,platform='win32',windows_api=lambda:(k,p))
    assert not p.GetProcessMemoryInfo.calls and not k.CloseHandle.calls


@pytest.mark.parametrize('kw',[{'sample_ok':False},{'rss':0}])
def test_windows_sample_refuses_missing_data(kw):
    k,p=api(**kw);r=pm.ResidentMemory(7,platform='win32',windows_api=lambda:(k,p))
    with pytest.raises(RuntimeError):r.sample()
    r.close();assert len(k.CloseHandle.calls)==1


def test_windows_close_failure_not_silent():
    k,p=api(close_ok=False);r=pm.ResidentMemory(7,platform='win32',windows_api=lambda:(k,p))
    with pytest.raises(RuntimeError,match='close failed'):r.close()
    r.close();assert len(k.CloseHandle.calls)==1


@pytest.mark.parametrize('pid',[0,-1,True,1.5,2**32])
def test_invalid_pid(pid):
    with pytest.raises(ValueError):pm.ResidentMemory(pid)


def test_unsupported_platform():
    with pytest.raises(RuntimeError,match='unsupported'):pm.ResidentMemory(7,platform='darwin')


@pytest.mark.parametrize('text',['Name: worker\n','VmRSS: nope kB\n','VmRSS: 2 MB\n','VmRSS: 0 kB\n'])
def test_linux_missing_or_invalid_fails_closed(monkeypatch,text):
    monkeypatch.setattr(pm.Path,'read_text',lambda self:text)
    r=pm.ResidentMemory(7,platform='linux')
    with pytest.raises((RuntimeError,ValueError)):r.sample()
    r.close()


def test_linux_bytes_and_closed(monkeypatch):
    monkeypatch.setattr(pm.Path,'read_text',lambda self:'VmRSS: 1536 kB\n')
    r=pm.ResidentMemory(7,platform='linux');assert r.sample()==1572864;r.close()
    with pytest.raises(RuntimeError):r.sample()


@pytest.mark.skipif(not sys.platform.startswith('linux'),reason='real proc check requires Linux')
def test_live_linux_self():
    r=pm.ResidentMemory(os.getpid());assert r.sample()>0;r.close()


class Child:
    pid=101
    def __init__(self,stdout,stderr,*,done=False,code=0,stubborn=False):
        self.done=done;self.code=code;self.returncode=code if done else None;self.stubborn=stubborn
        self.terminated=0;self.killed=0;self.waited=0
        stdout.write('{"ok":true}');stdout.flush();stderr.write('fixture stderr');stderr.flush()
    def poll(self):
        if self.done:self.returncode=self.code
        return self.returncode
    def terminate(self):
        self.terminated+=1
        if not self.stubborn:self.done=True;self.returncode=-15
    def wait(self,timeout=None):
        self.waited+=1
        if not self.done:raise subprocess.TimeoutExpired('test',timeout)
        return self.returncode
    def kill(self):self.killed+=1;self.done=True;self.returncode=-9


def setup(monkeypatch,*,mode='ok',stubborn=False,initial_done=False,code=0):
    children=[];monitors=[]
    class Monitor:
        def __init__(self,pid):
            self.pid=pid;self.closed=False;monitors.append(self)
            if mode=='open_fail' and pid==101:raise OSError('denied')
            if mode=='preflight_fail' and pid!=101:raise OSError('denied')
        def sample(self):
            if self.pid!=101:return 1
            if mode=='sample_fail':raise OSError('lost')
            if mode=='exit_race':children[0].done=True;children[0].returncode=0;raise OSError('exited')
            return 1500*1024*1024+1 if mode=='over' else 1
        def close(self):self.closed=True
    def popen(cmd,**kw):
        c=Child(kw['stdout'],kw['stderr'],done=initial_done,code=code,stubborn=stubborn);children.append(c);return c
    monkeypatch.setattr(pm,'ResidentMemory',Monitor);monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(m.time,'sleep',lambda seconds:(setattr(children[0],'done',True),setattr(children[0],'returncode',code)))
    return children,monitors


@pytest.mark.parametrize('mode',['over','open_fail','sample_fail'])
def test_guard_failure_stops_and_reaps(monkeypatch,mode):
    children,monitors=setup(monkeypatch,mode=mode)
    with pytest.raises(RuntimeError,match='not accepted'):m.guarded_file('s','a','o')
    assert children[0].terminated==1 and children[0].waited==(2 if mode=='sample_fail' else 1)
    assert all(q.closed for q in monitors if not(mode=='open_fail' and q.pid==101))


def test_guard_preflight_no_child(monkeypatch):
    children,_=setup(monkeypatch,mode='preflight_fail')
    with pytest.raises(OSError):m.guarded_file('s','a','o')
    assert not children


def test_guard_success_closes(monkeypatch):
    children,monitors=setup(monkeypatch)
    assert m.guarded_file('s','a','o')=={'ok':True}
    assert all(q.closed for q in monitors) and not children[0].terminated


def test_guard_exit_sampling_race(monkeypatch):
    _,monitors=setup(monkeypatch,mode='exit_race')
    assert m.guarded_file('s','a','o')=={'ok':True}
    assert all(q.closed for q in monitors)


def test_guard_fast_exit_does_not_need_child_handle(monkeypatch):
    _,monitors=setup(monkeypatch,initial_done=True)
    assert m.guarded_file('s','a','o')=={'ok':True}
    assert len(monitors)==1


def test_guard_nonzero_rejected(monkeypatch):
    setup(monkeypatch,initial_done=True,code=2)
    with pytest.raises(RuntimeError,match='fixture stderr'):m.guarded_file('s','a','o')


def test_timeout_kills_stubborn_worker_and_closes(monkeypatch):
    children,monitors=setup(monkeypatch,stubborn=True)
    ticks=iter([0,91]);monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
    with pytest.raises(RuntimeError,match='90second'):m.guarded_file('s','a','o')
    assert children[0].terminated==1 and children[0].killed==1 and children[0].waited==2
    assert all(q.closed for q in monitors)


def test_unexpected_exception_still_reaps(monkeypatch):
    children,monitors=setup(monkeypatch)
    monkeypatch.setattr(m.time,'sleep',lambda s:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):m.guarded_file('s','a','o')
    assert children[0].terminated and all(q.closed for q in monitors)


def test_completed_worker_over_time_is_not_accepted(monkeypatch):
    setup(monkeypatch,initial_done=True)
    ticks=iter([0,91]);monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
    with pytest.raises(RuntimeError,match='90second'):m.guarded_file('s','a','o')


def test_real_guard_small_guided_subprocess(tmp_path):
    import numpy as np
    from PIL import Image
    a=np.zeros((40,40),np.uint8);a[8:32,8:32]=255
    rgb=np.full((40,40,3),120,np.uint8)
    Image.fromarray(rgb).save(tmp_path/'s.png');Image.fromarray(a).save(tmp_path/'a.png')
    result=m.guarded_file(tmp_path/'s.png',tmp_path/'a.png',tmp_path/'out',method='guided')
    assert result['source_rgb_unchanged'] and result['quality']=='NOT_EVALUATED'
    assert (tmp_path/'out/matting-manifest.json').exists()


@pytest.mark.parametrize('code',[0,2,-9])
def test_missing_rss_during_teardown_bounded_reap(monkeypatch,code):
    children,monitors=setup(monkeypatch,mode='sample_fail')
    original=subprocess.Popen
    def popen(*args,**kw):
        child=original(*args,**kw)
        def wait(timeout=None):
            assert timeout==.2
            child.waited+=1;child.done=True;child.returncode=code;return code
        child.wait=wait
        return child
    monkeypatch.setattr(subprocess,'Popen',popen)
    if code:
        with pytest.raises(RuntimeError,match='fixture stderr'):m.guarded_file('s','a','o')
    else:
        assert m.guarded_file('s','a','o')=={'ok':True,'worker_guard_exit_reap_wait_seconds':.2}
    assert children[0].waited==1 and not children[0].terminated
    assert all(q.closed for q in monitors)
