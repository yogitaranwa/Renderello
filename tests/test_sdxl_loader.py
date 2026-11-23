"""No-download loader contract, not a model quality test."""
import sys
import types
from renderello.scene_providers import SDXLLightningProvider


def test_published_four_step_checkpoint_and_cache(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="fp16"))
    class UNet:
        @classmethod
        def from_config(cls, model, **kw):
            calls.append(("config", model, kw)); return cls()
        def to(self, *args): return self
        def load_state_dict(self, weights):
            assert weights == {"weights": "fake"}
    class Pipe:
        scheduler = types.SimpleNamespace(config={"old": 1})
        @classmethod
        def from_pretrained(cls, model, **kw):
            calls.append(("pipeline", model, kw)); return cls()
        def to(self, device): calls.append(("device", device)); return self
    class Scheduler:
        @classmethod
        def from_config(cls, config, **kw):
            calls.append(("scheduler", config, kw)); return cls()
    def download(**kw): calls.append(("download", kw)); return "/fake/checkpoint"
    monkeypatch.setitem(sys.modules,"diffusers",types.SimpleNamespace(
        StableDiffusionXLPipeline=Pipe,UNet2DConditionModel=UNet,EulerDiscreteScheduler=Scheduler))
    monkeypatch.setitem(sys.modules,"huggingface_hub",types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules,"safetensors.torch",types.SimpleNamespace(
        load_file=lambda path,device: {"weights":"fake"}))
    provider = SDXLLightningProvider()
    assert provider._load() is provider._load()
    dl = [x for x in calls if x[0]=="download"]
    assert len(dl)==1
    assert len(dl[0][1]["revision"]) == 40
    assert all(len(x[2]["revision"])==40 for x in calls if x[0] in ("pipeline","config"))
    assert dl[0][1]["filename"] == "sdxl_lightning_4step_unet.safetensors"
    assert any(x[0]=="scheduler" and x[2]["timestep_spacing"]=="trailing" for x in calls)


def test_notebook_installs_gpu_dependencies_and_compatible_pillow():
    from scripts.build_colab_notebook import build
    import json
    nb=json.loads(build().read_text())
    install="".join(nb["cells"][2]["source"])
    for package in ("diffusers","transformers","accelerate","safetensors","huggingface_hub"):
        assert package in install
    assert "pillow>=12.3,<13" in install


def test_runtime_version_capture_has_no_environment_secrets(monkeypatch):
    from renderello.studio import runtime_versions
    monkeypatch.setenv("MAGIC_HOUR_API_KEY", "not-a-real-key-test-only")
    versions = runtime_versions()
    assert versions["Pillow"]
    assert "MAGIC_HOUR_API_KEY" not in versions
    assert "not-a-real-key-test-only" not in str(versions)
