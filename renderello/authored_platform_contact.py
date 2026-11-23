"""Explicit authored platform planes and clipped 2D contact, not recovered 3D.

Opt-in native translation only. Supports two designed styles. A typed source
review supplies camera/material/contact spans; this module checks attachment
against the actual rasterized top-plane mask, never against a guessed floor.
"""
from dataclasses import replace
import numpy as np
from PIL import Image,ImageDraw
from .native_studio import background
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .contact_spans import eligible,ContactSpan,contact_map
from .candidate_harmonization import linear_over
from .checked_research.contact_camera import linear,srgb


def platform_scene(style,*,seed=0):
    if style not in ('clay_plinth','blush_steps'):raise ValueError('explicit clay/blush authored platform required')
    if type(seed) is not int or not 0<=seed<=2**32-1:raise ValueError('integer seed0..2**32-1 required')
    bg=background(style,seed=seed)
    # Clay uses its original visible top face. Blush gains an explicit top
    # trapezoid, replacing the ambiguous thin stripe only in this opt-in route.
    vertices=([(230,860),(760,860),(790,885),(205,885)] if style=='clay_plinth' else
              [(180,860),(844,860),(874,900),(150,900)])
    if style=='blush_steps':ImageDraw.Draw(bg).polygon(vertices,fill=(242,216,206))
    plane=Image.new('L',bg.size);ImageDraw.Draw(plane).polygon(vertices,fill=255)
    rgb=np.array(bg);mask=np.array(plane)>0
    return rgb,mask,dict(style=style,seed=seed,vertices=[list(v) for v in vertices],
        background_sha256=digest(rgb),surface_sha256=digest(mask),
        authored_surface=True,recovered_geometry=False,accepted=False)


def compose_platform_contact(source,alpha,style,spans,policy,*,translation,
        native_floor_row,support_tolerance=2,seed=0):
    native_rgb(source);native_alpha(alpha,source.shape[:2]);eligible(source,alpha,policy)
    if policy.camera!='front' or policy.source_pose!='front':raise ValueError('reviewed front camera/pose only; no viewpoint change')
    if not isinstance(translation,(list,tuple)) or len(translation)!=2 or any(type(v) is not int or v<0 for v in translation):
        raise ValueError('nonnegative integer translation required')
    if type(native_floor_row) is not int or not 0<=native_floor_row<alpha.shape[0]-1:
        raise ValueError('native support floor row required')
    if type(support_tolerance) is not int or not 0<=support_tolerance<=4:raise ValueError('integer0..4support tolerance required')
    if not isinstance(spans,(list,tuple)) or not spans or any(not isinstance(s,ContactSpan) for s in spans):
        raise ValueError('explicit typed native contact spans required')
    bg,plane,design=platform_scene(style,seed=seed);dx,dy=translation;h,w=alpha.shape
    if dx+w>1024 or dy+h>1024:raise ValueError('entire native source must fit; no crop/resize')
    # Validate native bounds before translating, so blank canvas cannot excuse
    # a declaration that lies outside the original image.
    if any(s.x1>w or s.bottom_max>=h for s in spans):raise ValueError('native span outside source')
    rgb=np.zeros_like(bg);a=np.zeros((1024,1024),np.uint8)
    rgb[dy:dy+h,dx:dx+w]=source;a[dy:dy+h,dx:dx+w]=alpha
    canvas_policy=replace(policy,source_sha256=digest(rgb),alpha_sha256=digest(a))
    canvas_spans=[ContactSpan(s.name,s.x0+dx,s.x1+dx,s.bottom_min+dy,s.bottom_max+dy) for s in spans]
    shade,contact=contact_map(rgb,a,canvas_spans,canvas_policy)
    hard=a>=128;lower=np.where(hard.any(0),1023-np.argmax(hard[::-1],axis=0),-1);attached=[]
    for span in canvas_spans:
        cols=np.arange(span.x0,span.x1);rows=lower[cols]
        if np.any(np.abs(rows-(native_floor_row+dy))>support_tolerance):
            raise ValueError('declared floor contradicts observed native contact contour')
        if not np.all(plane[rows,cols]&plane[rows+1,cols]):
            raise ValueError('actual contact footprint lies outside visible top plane')
        attached.append(dict(name=span.name,columns=len(cols),all_first_rows_on_surface=True))
    clipped=shade.copy();clipped[~plane]=0
    for span in canvas_spans:
        cols=np.arange(span.x0,span.x1)
        if not np.all(clipped[lower[cols]+1,cols]>0):raise ValueError('clipped shadow detached from contact')
    out=linear_over(rgb,a,srgb(linear(bg)*(1-clipped[...,None])))
    if np.any(clipped[hard]) or not np.array_equal(out[a==255],rgb[a==255]):raise RuntimeError('source/shadow invariant failed')
    record=dict(schema='authored-platform-contact.v1',design=design,contact=contact,
        source_sha256=digest(source),alpha_sha256=digest(alpha),
        source_to_canvas_translation=list(translation),native_floor_row=native_floor_row,
        support_tolerance=support_tolerance,attachments=attached,
        shadow_pixels_clipped_off_surface=int(((shade>0)&~plane).sum()),
        shadow_pixels_outside_surface=int(((clipped>0)&~plane).sum()),
        result_sha256=digest(out),shadow_sha256=digest(clipped),source_cropped=False,
        source_resampled=False,source_rgb_repainted=False,opaque_source_pixels_exact=True,
        accepted=False,independent_review=False,recovered_geometry=False,physical_relighting=False,
        limitations=['Visible platform plane is authored 2D design, not recovered geometry.',
          'Reviewed camera, material, completeness and support spans are not independently verified.',
          'Source lighting, edge contamination and old reflections remain unchanged.',
          'Raster attachment and source preservation do not imply plausible lighting or publication quality.'])
    return out,clipped,bg,plane,record
