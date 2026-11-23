"""Recorded executor class is not a quality, identity or spend attestation."""


def describe_execution(route, backend='', *, scene_mock=False, reference=False):
    if backend=='stub' and route in ('pure_prompt','hybrid'):
        kind='TEST_DOUBLE_NOT_GENERATIVE_MODEL'
        caption='TEST DOUBLE: deterministic test editor, not an AI product edit or model-quality result.'
    elif isinstance(backend,str) and backend.startswith('magichour:') and backend.endswith(':mock'):
        kind='TEST_DOUBLE_NOT_GENERATIVE_MODEL'
        caption='MOCK MAGIC HOUR: deterministic offline image echo, not a model-quality result or actual charge.'
    elif isinstance(backend,str) and backend.startswith('magichour:') and backend.endswith(':live'):
        kind='RECORDED_LIVE_EDITOR_CONFIGURATION'
        caption='Recorded live Magic Hour configuration; inspect provider receipts for execution/charges and images for quality.'
    elif route=='scene_compose' and scene_mock:
        kind='PROCEDURAL_MOCK_SCENE'
        caption='MOCK SCENE: procedural gradient, not a generated photoshoot or real relighting.'
    elif route=='scene_compose' and reference:
        kind='SAVED_REFERENCE_BACKDROP'
        caption='SAVED BACKDROP: uploaded image compositing, no new scene-generation model.'
    elif route=='cutout_compose':
        kind='DETERMINISTIC_COMPOSITE'
        caption='DETERMINISTIC COMPOSITE: source alpha pasted onto a backdrop; not new product generation.'
    else:
        kind='EXECUTOR_NOT_CLASSIFIED'
        caption='Execution type not classified here. Consult the recorded backend and run evidence.'
    return dict(kind=kind,caption=caption,quality_evidence=False,credits_verified=False)
