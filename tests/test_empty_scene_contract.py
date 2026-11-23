from dataclasses import replace
from pathlib import Path
import re
import pytest
from renderello.templates import TEMPLATES, get_template
from renderello.empty_scene_prompt import build_empty_scene_prompt, UnsupportedEmptySceneStaging

SUPPORTED = ['clean_studio','lifestyle_kitchen','natural_outdoor','seasonal_holiday']
UNSUPPORTED = ['luxury_hero','editorial_beauty','minimal_tabletop','social_ad','reflective_surface','flat_lay']

@pytest.mark.parametrize('key',SUPPORTED)
def test_positive_fragments_have_no_foreground_or_equipment(key):
    p=build_empty_scene_prompt(get_template(key))
    positive=re.split(r'Completely empty (?:set|foreground):',p)[0]
    body=positive.split('Set:',1)[1]
    assert not re.search(r'\b(product|bottle|label|logo|packaging|softbox|reflector|tripod|pedestal|podium|plinth)\b|fill card',body,re.I)
    assert 'occupying 70%' not in p and 'product centered' not in p
    assert 'no product' in p and len(p)>900

@pytest.mark.parametrize('key',UNSUPPORTED)
def test_incompatible_guide_does_not_become_generic_gray_floor(key):
    with pytest.raises(UnsupportedEmptySceneStaging,match='unsupported'):
        build_empty_scene_prompt(get_template(key))

@pytest.mark.parametrize('key',list(TEMPLATES))
def test_native_template_was_not_rewritten(key):
    before=repr(get_template(key))
    try:build_empty_scene_prompt(get_template(key))
    except UnsupportedEmptySceneStaging:pass
    assert repr(get_template(key))==before

def test_seasonal_distant_decoration_is_not_globally_forbidden():
    p=build_empty_scene_prompt(get_template('seasonal_holiday'))
    assert 'distant pine foliage' in p and 'no props' not in p
    assert 'central placement area and all foreground remain bare' in p

def test_landscape_ratio_is_not_lost():
    assert 'Wide 3:2' in build_empty_scene_prompt(get_template('clean_studio'),aspect_key='3:2')

def test_modified_known_template_requires_explicit_new_environment_contract():
    with pytest.raises(ValueError,match='modified'):
        build_empty_scene_prompt(replace(get_template('clean_studio'),scene='bottle in a kitchen'))

def test_unregistered_foreground_or_equipment_cannot_bypass_contract():
    class Custom:
        scene='a bottle on a neutral tabletop'
        lighting='soft light'
        camera='level view'
        mood='quiet'
    with pytest.raises(ValueError,match='positive empty-scene'):
        build_empty_scene_prompt(Custom())

def test_unsupported_staging_refused_before_reservation_or_backend(tmp_path,monkeypatch):
    import renderello.studio as studio
    import renderello.backends.magichour as mh
    from renderello.presets import get_preset
    from renderello.schemas import Aspect
    from types import SimpleNamespace
    monkeypatch.setenv('RENDERELLO_MH_LIVE','1')
    def no(*args,**kwargs):raise AssertionError('paid or reservation path reached')
    monkeypatch.setattr(studio,'reserve_budget',no)
    monkeypatch.setattr(mh,'MagicHourBackend',no)
    preset=next(p for p in __import__('renderello.presets',fromlist=['PRESETS']).PRESETS.values() if p.template_key=='minimal_tabletop')
    req=SimpleNamespace(knob_values={},variants=1,instruction='')
    with pytest.raises(UnsupportedEmptySceneStaging):
        studio._run_magichour_scene(req,preset,Aspect.SQUARE,'test',tmp_path,None,'',[],[],0)
