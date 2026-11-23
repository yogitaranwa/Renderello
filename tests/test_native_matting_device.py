import pytest
from renderello.checked_research.native_matting import CpuPredictor


@pytest.mark.parametrize('value',['gpu','cuda:0','auto',None,1])
def test_reject_implicit_or_unhandled_device(value):
    with pytest.raises(ValueError,match='explicit cpu or cuda'):
        CpuPredictor(device=value)


def test_cuda_refuses_without_gpu_before_download(monkeypatch):
    import sys,types
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
    monkeypatch.setitem(sys.modules,'torch',torch)
    with pytest.raises(RuntimeError,match='no CPU fallback'):
        CpuPredictor(device='cuda')
