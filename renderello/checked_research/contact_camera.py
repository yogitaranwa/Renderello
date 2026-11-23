"""Reviewer-declared camera/support study, not inferred geometry or glass optics.

No foreground resampling or colour changes. Scene illumination is compressed to
an explicitly designed smooth field to prevent invented props in the final plate.
The unmodified generated plate is retained as evidence. Production flags fail
closed; they are not an automatic acceptance classifier.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, label


@dataclass(frozen=True)
class SceneSpec:
    camera: str
    light: str = 'left'
    transparent: bool = False
    cropped: bool = False
    multiobject: bool = False
    support_reviewed: bool = False

    def __post_init__(self):
        if self.camera not in ('front', 'elevated', 'flatlay'):
            raise ValueError('explicit front/elevated/flatlay camera required')
        if self.light not in ('left', 'right', 'diffuse'):
            raise ValueError('explicit light direction required')
        if any(type(getattr(self, k)) is not bool for k in
               ('transparent', 'cropped', 'multiobject', 'support_reviewed')):
            raise ValueError('boolean review declarations required')


def _image(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError('uint8 RGB image required')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError('nontrivial RGB image required')
    if rgb.shape[0]*rgb.shape[1] > 8_000_000:
        raise ValueError('image exceeds research allocation limit')


def linear(rgb):
    x = np.asarray(rgb, dtype=np.float64)/255.
    return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)


def srgb(x):
    x = np.clip(x, 0., 1.)
    return np.rint(np.where(x <= .0031308, x*12.92,
                           1.055*x**(1/2.4)-.055)*255).astype(np.uint8)


def layout(alpha, occupancy=.72):
    """Crop empty source margins only. Native pixels retained; integer translation."""
    if (not isinstance(alpha, np.ndarray) or alpha.dtype != np.uint8
            or alpha.ndim != 2 or min(alpha.shape) < 2):
        raise ValueError('native uint8 alpha required')
    if alpha.size > 4_000_000 or not np.any(alpha >= 128):
        raise ValueError('nonempty bounded opaque support required')
    if isinstance(occupancy, bool) or not np.isfinite(occupancy) or not .4 <= occupancy <= .8:
        raise ValueError('occupancy must be .4 through .8')
    yy, xx = np.nonzero(alpha > 0)
    box = (int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1))
    width, height = box[2]-box[0], box[3]-box[1]
    side = max(256, int(np.ceil(max(width, height)/occupancy)))
    if side > 2400:
        raise ValueError('native composition exceeds allocation limit')
    left = (side-width)//2
    top = int((side-height)*.52)
    return box, (side, side), (left, top)


def smooth_scene(raw, shape, spec):
    """Robust quadratic illumination field; no generated object texture survives.

    This intentionally simplifies the scene. It is not a generated photograph or
    automatic semantic prop detection, and cannot certify source-light matching.
    """
    from PIL import Image
    _image(raw)
    h, w = shape
    if h < 2 or w < 2 or h*w > 6_000_000:
        raise ValueError('bounded target dimensions required')
    sample = np.asarray(Image.fromarray(raw).resize((32, 32))).astype(float)/255.
    sy, sx = np.mgrid[-1:1:32j, -1:1:32j]
    basis = np.stack([np.ones_like(sx), sx, sy, sx*sy, sx*sx, sy*sy], -1)
    X = basis.reshape(-1, 6); Y = sample.reshape(-1, 3)
    # Huber-style reweighting reduces local generated props before low-order fit.
    weight = np.ones(len(X))
    for _ in range(5):
        co = np.linalg.lstsq(X*weight[:, None], Y*weight[:, None], rcond=None)[0]
        residual = np.linalg.norm(X@co-Y, axis=1)
        scale = max(.01, float(np.median(residual))*1.4826)
        weight = np.minimum(1., 1.5*scale/np.maximum(residual, 1e-6))
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    B = np.stack([np.ones_like(x), x, y, x*y, x*x, y*y], -1)
    field = B@co
    # Keep a restrained commercial studio field instead of large fitted blobs.
    mean = np.median(Y, axis=0)
    field = mean + .38*(field-mean)
    lateral = (x if spec.light == 'right' else -x) if spec.light != 'diffuse' else 0*x
    field += .025*lateral[..., None]
    if spec.camera != 'flatlay':
        # Smooth continuous floor, deliberately no horizon/ledge to float above.
        field -= .025*np.maximum(y, 0)[..., None]
    return np.rint(np.clip(field, .08, .96)*255).astype(np.uint8)


def shadow_map(alpha, shape, position, spec):
    """Flatlay full silhouette or upright lower-contact contour plus cast shadow.

    Contact starts at the actual native lower contour. No fabricated 3D pose.
    Upright cast uses a declared 2D height projection, never measured geometry.
    """
    h, w = shape; x, y = position
    ah, aw = alpha.shape
    if x < 0 or y < 0 or x+aw > w or y+ah > h:
        raise ValueError('native alpha must fit canvas')
    a = alpha.astype(float)/255.; direction = 1 if spec.light == 'left' else -1
    if spec.camera == 'flatlay':
        full = np.zeros(shape); full[y:y+ah, x:x+aw] = a
        offset = max(2, round(max(aw, ah)*.009))
        from scipy.ndimage import shift
        cast = shift(full, (offset, direction*offset), order=1, mode='constant')
        tight = gaussian_filter(cast, max(1.2, max(aw, ah)*.003))*.24
        broad = gaussian_filter(cast, max(3., max(aw, ah)*.014))*.12
        return np.clip(1-(1-tight)*(1-broad), 0, .45)
    support = alpha >= 128
    bottoms = np.full(aw, -1, dtype=int)
    for col in np.flatnonzero(support.any(axis=0)):
        bottoms[col] = np.flatnonzero(support[:, col])[-1]
    if not np.any(bottoms >= 0):
        raise ValueError('nonempty support required for upright shadow')
    floor = int(bottoms.max())
    span = np.flatnonzero(bottoms >= floor-max(2, round(ah*.035)))
    # The previous blurred one-pixel line lost contact density and its projected
    # mass sat below the object. Build an attached footprint with a finite-width
    # core, then a rapidly decaying tail. The first uncovered pixel is strongest.
    contact = np.zeros(shape)
    reach = max(4, round(ah*.025))
    decay = max(1.2, ah*.004)
    for col in span:
        row = y+bottoms[col]+1
        length = min(reach, h-row)
        if length > 0:
            contact[row:row+length, x+col] = .62*np.exp(-np.arange(length)/decay)
    # Horizontal-only smoothing cannot move the footprint away from its base.
    from scipy.ndimage import gaussian_filter1d
    contact = gaussian_filter1d(contact, max(.6, aw*.0015), axis=1)
    # A broad tail must start on the same support, not at an offset ellipse.
    tail = gaussian_filter(contact, (max(2., ah*.008), max(2., aw*.012)))*.45
    # Shadow length is an explicit 2D assumption. Compress toward the attached
    # support plane and attenuate with height, avoiding a second detached island.
    cast = np.zeros(shape)
    for row in range(ah):
        height = max(0, floor-row)/max(1, floor)
        cols = np.flatnonzero(a[row] > 0)
        xx = x+cols+np.rint(direction*height*ah*.12).astype(int)
        yy = np.full(len(cols), y+floor+round(height*ah*.025), dtype=int)
        valid = (xx >= 0)&(xx < w)&(yy >= 0)&(yy < h)
        np.maximum.at(cast, (yy[valid], xx[valid]), a[row, cols[valid]]*.10*(1-height*.7))
    cast = gaussian_filter(cast, max(1.2, aw*.006))
    return np.clip(1-(1-contact)*(1-tail)*(1-cast), 0, .68)


def composite(source, alpha, raw_scene, spec):
    _image(source); _image(raw_scene)
    if not isinstance(spec, SceneSpec): raise ValueError('SceneSpec required')
    if alpha.shape != source.shape[:2]: raise ValueError('source/alpha pairing required')
    box, (w, h), position = layout(alpha)
    x0, y0, x1, y1 = box
    rgb = source[y0:y1, x0:x1]; a = alpha[y0:y1, x0:x1]
    bg = smooth_scene(raw_scene, (h, w), spec)
    shade = shadow_map(a, (h, w), position, spec)
    bg_shadow = srgb(linear(bg)*(1-shade[..., None]))
    x, y = position; ph, pw = a.shape
    out = bg_shadow.copy(); dst = out[y:y+ph, x:x+pw]
    weight = a[..., None].astype(float)/255.
    dst[:] = srgb(linear(rgb)*weight+linear(dst)*(1-weight))
    dst[a == 255] = rgb[a == 255]
    _, number = label(alpha >= 128)
    flags = []
    if spec.transparent: flags.append('transparent-material-optics-not-recovered')
    if spec.cropped: flags.append('source-cropped-product-not-recovered')
    if spec.multiobject: flags.append('multi-object-membership-and-ghosts-need-review')
    if not spec.support_reviewed: flags.append('physical-support-not-reviewed')
    if number > 30: flags.append('fragmented-mask-needs-review')
    return out, bg_shadow, shade, dict(
        sourceCrop=list(box), nativePlacement=list(position), canvas=[w, h],
        foregroundResampled=False, changedOpaquePixels=int(np.any(dst[a == 255] != rgb[a == 255], axis=1).sum()),
        opaquePixels=int((a == 255).sum()), camera=spec.camera, light=spec.light,
        shadowGeometry='attached-footprint-declared-2D-approximation', sceneRepresentation='robust-quadratic-generated-illumination-field',
        sourceRelit=False, glassOpticsRecovered=False, semanticPropDetection=False,
        alphaUnchanged=True, maskComponents=int(number), unresolvedFlags=flags,
        commercialAcceptance=None, acceptanceRequiresVisualReview=True)
