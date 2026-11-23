"""Explicit constrained template prompts, not the original shoot specification.

No model execution, mask recovery, viewpoint/light inference or acceptance.
The caller's source declaration is a hypothesis and never authenticates a photo.
"""
from dataclasses import asdict
from .schemas import EditRequest
from .source_contract import SourceContract
from .prompts import FIDELITY_GUARD, validate_instruction

# Environment-only alternatives. Original camera/light/rotation/reflection
# demands are intentionally not carried into this constrained experiment.
CONTEXTS = {
 'clean_studio':'a plain neutral-white studio background with uncluttered negative space',
 'luxury_hero':'a dark charcoal premium studio background, without a generated product reflection',
 'editorial_beauty':'a quiet pastel beige editorial background with distant soft botanical accents',
 'lifestyle_kitchen':'a softly defocused warm kitchen environment behind the existing product',
 'natural_outdoor':'a defocused natural green outdoor environment behind the existing product',
 'minimal_tabletop':'a simple light-gray background without a newly invented podium under the product',
 'seasonal_holiday':'a warm seasonal background with restrained distant bokeh and no product occlusion',
 'social_ad':'a bold single-colour graphic background with copy space, no added text and no product rotation or floating',
 'reflective_surface':'a dark premium background without inventing a new reflection of the product',
 'flat_lay':'an uncluttered neutral background, retaining the existing source viewpoint rather than demanding overhead',
}


def build_source_preserving_prompt(request: EditRequest, declaration: SourceContract):
    if not isinstance(request,EditRequest) or not isinstance(declaration,SourceContract):
        raise ValueError('typed request and explicit source declaration required')
    if declaration.pose=='unknown' or declaration.material=='unknown' or declaration.framing=='unknown':
        raise ValueError('explicit pose/material/framing declarations required; no inference')
    if declaration.material!='opaque':
        raise ValueError('opaque-only constrained experiment; no glass/transmission recovery')
    if declaration.framing!='complete':
        raise ValueError('complete source required; no uncropping or missing-part invention')
    key=request.template.key
    if key not in CONTEXTS:raise ValueError('registered constrained context required')
    instruction=validate_instruction(request.custom_instruction).strip()
    parts=[FIDELITY_GUARD,
       'This is a source-preserving background-context experiment, not a new camera or product-lighting setup.',
       f'Retain the photographed product viewpoint and pose (caller declaration: {declaration.pose}); do not rotate, tilt, reshape, uncrop or reconstruct it.',
       'Retain the existing product illumination, highlights, colour and material. Do not relight or repaint the product to fit the background.',
       f'Change only the surrounding environment to {CONTEXTS[key]}.',
       'Do not add or remove products, source-visible parts, product reflections or cast shadows that would require guessing hidden geometry. Target support and contact still need review.']
    if instruction:parts.append('Additional environment-only instruction: '+instruction+'.')
    return dict(schema='renderello.constrained-context-prompt.v1',prompt=' '.join(parts),
        base_template=asdict(request.template),variant='EXPLICIT_SOURCE_PRESERVING_CONTEXT_NOT_ORIGINAL_SHOOT',
        source_declaration=asdict(declaration),declaration_provenance='CALLER_HYPOTHESIS_NOT_CAPTURE_TRUTH',
        original_requirements_not_fulfilled=['original_scene_specification','original_camera_specification','original_lighting_specification','requested_product_rotation_or_new_reflection'],
        source_pixel_lock_guaranteed=False,model_obedience_verified=False,alpha_verified=False,
        target_contact_verified=False,polished_template=False,photo_accepted=False,
        executed=False,paid_call_authorized=False,default_pipeline_changed=False)


def main(argv=None):
    """Write an explicit executable prompt-preparation artifact, never generate."""
    import argparse, json, hashlib
    from pathlib import Path
    from .templates import get_template
    from .pipeline_intake import load_generation_image
    parser=argparse.ArgumentParser(description='Prepare a constrained source-preserving context prompt. No image generation or spending.')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--template',required=True,choices=tuple(CONTEXTS))
    parser.add_argument('--source-pose',required=True,choices=('front','elevated','overhead','suspended'))
    parser.add_argument('--material',required=True,choices=('opaque',))
    parser.add_argument('--framing',required=True,choices=('complete',))
    parser.add_argument('--instruction',default='')
    parser.add_argument('--allow-colour-conversion',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():raise FileExistsError('new prompt artifact required; never replace earlier work')
    request=EditRequest(args.source,get_template(args.template),custom_instruction=args.instruction)
    record=build_source_preserving_prompt(request,SourceContract(args.source_pose,args.material,args.framing))
    image,intake=load_generation_image(args.source,allow_colour_conversion=args.allow_colour_conversion)
    image.close()
    record['image_intake']=intake
    record['source_file_sha256']=intake['source_sha256']
    record['benchmark_role']='SEPARATE_CONSTRAINED_EXPERIMENT_CANNOT_REPLACE_ORIGINAL_TEMPLATE_CASE'
    record['generation_authority']='NONE_PROMPT_PREPARATION_ONLY'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x')as f:json.dump(record,f,indent=2,default=str)
    print(str(args.out))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
