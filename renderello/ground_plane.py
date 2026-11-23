"""Explicit studio foreground plane experiment, not inferred physical geometry.

A designer chooses horizon and color. This replaces scene foreground pixels,
never moves the saved product, and never claims to detect a real tabletop.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .compose import finish_scene


def studio_foreground(scene, crop, placement, *, horizon, color):
    if scene.dtype != np.uint8 or scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError('scene must be RGB uint8')
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 4:
        raise ValueError('product must be RGBA uint8')
    h,w=scene.shape[:2];x,y=map(int,placement);ph,pw=crop.shape[:2]
    if x<0 or y<0 or x+pw>w or y+ph>h:raise ValueError('product clips canvas')
    rows=np.where((crop[...,3]>127).any(axis=1))[0]
    if not len(rows):raise ValueError('empty product')
    foot=y+int(rows[-1])
    if not isinstance(horizon,int) or not 0<=horizon<foot:raise ValueError('horizon must be above foot')
    rgb=np.asarray(color,dtype=float)
    if rgb.shape!=(3,) or not np.isfinite(rgb).all() or (rgb<0).any() or (rgb>255).any():
        raise ValueError('color must contain three finite values0-255')
    out=scene.copy()
    # A mild near-to-far tonal gradient gives a deliberately designed studio
    # surface. There is no generated texture or hidden inpainting here.
    t=np.linspace(0,1,h-horizon)[:,None,None]
    plane=np.clip(rgb[None,None,:]*(1.02-.13*t),0,255)
    out[horizon:]=np.broadcast_to(plane,(h-horizon,w,3)).astype(np.uint8)
    return out,{'mode':'explicit_designed_foreground','horizon':horizon,
                'foot_row':foot,'color':rgb.tolist(),'automatic_geometry':False}


def replay_ground_plane(cache, *, output, horizon, color):
    cache=Path(cache);output=Path(output)
    if output.exists():raise ValueError('output exists; choose a new path')
    scene=np.array(Image.open(cache/'scene.png').convert('RGB'))
    crop=np.array(Image.open(cache/'product_rgba.png').convert('RGBA'))
    m=json.loads((cache/'layers.json').read_text());place=tuple(m['placement'])
    if m.get('cover_used'):raise ValueError('cover placement unsupported')
    plane,diag=studio_foreground(scene,crop,place,horizon=horizon,color=color)
    final,_=finish_scene(plane,crop,place,shadow_style='anchored soft',
                        glass_backdrop=m.get('glass_backdrop'),scene_auto_lift=False)
    result=np.clip(final,0,255).astype(np.uint8)
    x,y=place;ph,pw=crop.shape[:2];opaque=crop[...,3]==255
    diag['opaque_rgb_exact']=None if m.get('glass_backdrop') is not None else bool(
        np.array_equal(result[y:y+ph,x:x+pw][opaque],crop[...,:3][opaque]))
    if diag['opaque_rgb_exact'] is False:raise ValueError('product fidelity invariant failed')
    output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(result).save(output)
    output.with_suffix('.json').write_text(json.dumps(diag,indent=2))
    return diag
