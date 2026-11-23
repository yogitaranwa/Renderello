"""Declared scene-only exposure/color matching, not physical relighting.

Transforms only background pixels before native product composition. No light,
material or camera estimation is claimed. Every setting is explicit and saved.
"""
import numpy as np


def finish_background(rgb, *, exposure_ev=0., white_balance=(1.,1.,1.)):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('background must be RGB uint8')
    if not np.isfinite(exposure_ev) or abs(exposure_ev)>.5:
        raise ValueError('scene exposure must be finite and within half a stop')
    wb=np.asarray(white_balance,dtype=np.float64)
    if wb.shape!=(3,) or not np.isfinite(wb).all() or (wb<.85).any() or (wb>1.15).any():
        raise ValueError('declared white balance gains must be finite RGB0.85to1.15')
    if exposure_ev==0 and (wb==1).all():return rgb.copy()
    srgb=rgb.astype(np.float64)/255
    linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    linear=np.clip(linear*(2**exposure_ev)*wb,0,1)
    result=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    return np.rint(np.clip(result,0,1)*255).astype(np.uint8)


def validate_support_plane(mask, alpha, placement):
    """Require all lower opaque contour samples on an explicitly reviewed plane.

    The mask is geometry supplied after inspecting the actual background,
    never inferred from its filename, generated prompt or product pixel audit.
    """
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.ndim!=2 or min(mask.shape)<1:
        raise ValueError('support plane must be uint8 L')
    if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.ndim!=2:
        raise ValueError('product alpha must be uint8 L')
    x,y=placement;h,w=mask.shape;ph,pw=alpha.shape
    if any(type(v) is not int for v in [x,y]) or x<0 or y<0 or x+pw>w or y+ph>h:
        raise ValueError('native placement outside support canvas')
    yy,xx=np.where(alpha>127)
    if not len(yy):raise ValueError('opaque support contour required')
    bottom=int(yy.max());band=max(2,round(ph*.025));samples=[]
    for col in range(pw):
        rows=np.flatnonzero(alpha[:,col]>127)
        if len(rows) and rows[-1]>=bottom-band:samples.append((y+int(rows[-1])+1,x+col))
    if not samples or any(row>=h or mask[row,col]!=255 for row,col in samples):
        raise ValueError('product support misses the reviewed support plane; reframe background or placement')
    return len(samples)


def inspect_support_components(alpha):
    """Refuse detached lower islands that silently set the contact baseline.

    Component area is not product identity. This detects ambiguity; it never
    deletes a watermark, removes a component or declares a semantic mask good.
    Intentionally disconnected products need a separately reviewed support path.
    """
    import cv2
    if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.ndim!=2:
        raise ValueError('product alpha must be uint8 L')
    n,labels,stats,centers=cv2.connectedComponentsWithStats((alpha>127).astype(np.uint8),connectivity=8)
    if n<2:raise ValueError('opaque support contour required')
    main=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    main_bottom=int(stats[main,cv2.CC_STAT_TOP]+stats[main,cv2.CC_STAT_HEIGHT]-1)
    ambiguous=[]
    for idx in range(1,n):
        if idx==main:continue
        area=int(stats[idx,cv2.CC_STAT_AREA]);bottom=int(stats[idx,cv2.CC_STAT_TOP]+stats[idx,cv2.CC_STAT_HEIGHT]-1)
        if bottom>main_bottom+3:
            ambiguous.append(dict(area=area,bottom=bottom,box=[int(v) for v in stats[idx,:4]]))
    report=dict(component_count=n-1,largest_component_area=int(stats[main,cv2.CC_STAT_AREA]),largest_component_bottom=main_bottom,lower_detached_components=ambiguous,mask_semantics='NOT_EVALUATED')
    if ambiguous:raise ValueError('detached lower mask component sets contact baseline; review source mask and intended support without silently deleting components')
    return report
