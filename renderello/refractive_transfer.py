"""Experimental clean-plate refractive-difference compositor, not an optics estimator.
Original implementation of a limited image formation model in linear sRGB.
No third-party network, dataset or implementation is bundled.
"""
from pathlib import Path
import json
import hashlib
from io import BytesIO
import numpy as np
from PIL import Image
from .intent_mask import sha

LIMITATIONS=[
 'Experimental transfer requires externally supplied inspected refractive flow and transmission, not inferred from a product photo.',
 'Single background-plane correspondence and fixed additive residual only; not full 3D refraction, changed reflections, caustics, relighting or pose.',
 'Clean plate must match source camera, framing, exposure and lighting. Hash checks bind files, not physical alignment.',
 'Transparent product pixels necessarily change with the background; only zero-transmission pixels are source-exact.',
 'Synthetic validation is not a real-glass quality pass.'
]


def decode(rgb):
 a=np.asarray(rgb,dtype=np.float64)/255
 return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)


def encode(a):
 a=np.asarray(a,dtype=np.float64)
 return np.rint(np.clip(np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055),0,1)*255).astype(np.uint8)


def warp(image,flow,support):
 h,w=image.shape[:2]
 if flow.shape!=(h,w,2) or not np.issubdtype(flow.dtype,np.floating) or not np.isfinite(flow).all():raise ValueError('finite floating HxWx2 flow required')
 yy,xx=np.mgrid[:h,:w];x=xx+flow[:,:,0];y=yy+flow[:,:,1]
 if np.any(support & ((x<0)|(x>w-1)|(y<0)|(y>h-1))):raise ValueError('refracted sampling outside background plane; no border fabrication')
 x=np.clip(x,0,w-1);y=np.clip(y,0,h-1);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);wx=(x-x0)[:,:,None];wy=(y-y0)[:,:,None]
 return (image[y0,x0]*(1-wx)+image[y0,x1]*wx)*(1-wy)+(image[y1,x0]*(1-wx)+image[y1,x1]*wx)*wy


def transfer_arrays(source,plate,target,flow,transmission,support):
 if source.shape!=plate.shape or source.shape!=target.shape or source.ndim!=3 or source.shape[2]!=3:raise ValueError('matching RGB images required')
 if source.dtype!=np.uint8 or plate.dtype!=np.uint8 or target.dtype!=np.uint8:raise ValueError('uint8 sRGB image arrays required')
 h,w=source.shape[:2]
 if transmission.shape!=(h,w,3) or not np.isfinite(transmission).all() or np.any((transmission<0)|(transmission>1)):raise ValueError('finite HxWx3 transmission in0..1 required')
 if support.shape!=(h,w) or support.dtype!=np.bool_ or not np.any(support):raise ValueError('nonempty boolean support required')
 src=decode(source);old=warp(decode(plate),flow,support);new=warp(decode(target),flow,support)
 residual=src-transmission*old
 # A negative radiance residual indicates incompatible plate/transmission under this model.
 if np.any(residual[support]<-.02):raise ValueError('negative source residual; plate/transmission incompatible with additive model')
 prediction=src+transmission*(new-old)
 clipped=np.any((prediction<0)|(prediction>1),axis=2)&support
 if np.any(clipped):raise ValueError('target exceeds linear radiance gamut; no silent clipping')
 result=target.copy();result[support]=encode(prediction[support]);protected=support & np.all(transmission==0,axis=2)
 result[protected]=source[protected]
 return result,dict(support_pixels=int(support.sum()),zero_transmission_pixels=int(protected.sum()),zero_transmission_rgb_exact=bool(np.array_equal(result[protected],source[protected])),out_of_gamut_pixels=0,linear_residual_min=float(residual[support].min()),flow_nonzero_pixels=int((support & np.any(flow!=0,axis=2)).sum()))


def _snapshot(path,limit):
 # Decode exactly the bytes whose hash was checked, not a later file read.
 with Path(path).open('rb') as stream:data=stream.read(limit+1)
 if len(data)>limit:raise ValueError('input exceeds bounded byte snapshot: '+Path(path).name)
 return data


def transfer_files(source,plate,target,flow,transmission,support,spec,out,*,confirm_assumptions=False):
 if not confirm_assumptions:raise ValueError('explicit limited-model confirmation required')
 if out.exists():raise ValueError('output exists; preserve previous experiment')
 paths=dict(source=source,plate=plate,target=target,flow=flow,transmission=transmission,support=support)
 spec_bytes=_snapshot(spec,64_000)
 d=json.loads(spec_bytes)
 if d.get('schema')!='renderello-refractive-transfer-v1' or d.get('color_space')!='srgb' or d.get('flow_convention')!='source_pixel_to_background_dxdy_pixels':raise ValueError('unsupported optics specification')
 if d.get('capture_alignment_reviewed') is not True or d.get('maps_reviewed') is not True:raise ValueError('capture and map review declarations required')
 if d.get('material')!='transparent' or d.get('framing')!='complete':raise ValueError('complete transparent source declaration required')
 if d.get('map_origin') not in ('synthetic_known','external_reviewed'):raise ValueError('explicit map origin required')
 snapshots={name:_snapshot(p,80_000_000 if name=='flow'else 32_000_000)for name,p in paths.items()}
 for name,data in snapshots.items():
  if hashlib.sha256(data).hexdigest()!=d.get('input_sha256',{}).get(name):raise ValueError('source-bound input hash mismatch: '+name)
 arrays={}
 for name in ('source','plate','target','transmission','support'):
  with Image.open(BytesIO(snapshots[name])) as im:
   if im.width*im.height>4_194_304:raise ValueError('image exceeds4MP experimental bound')
   if im.mode!=('L' if name=='support' else 'RGB'):raise ValueError('RGB images/transmission and L binary support required')
   arrays[name]=np.array(im)
 if d.get('image_size')!=[arrays['source'].shape[1],arrays['source'].shape[0]]:raise ValueError('source dimensions mismatch')
 if not np.isin(arrays['support'],[0,255]).all():raise ValueError('support must be binary0/255; transmission carries transparency')
 f=np.load(BytesIO(snapshots['flow']),allow_pickle=False)
 if not isinstance(f,np.ndarray):raise ValueError('plain NPY flow required')
 result,stats=transfer_arrays(arrays['source'],arrays['plate'],arrays['target'],f,arrays['transmission'].astype(float)/255,arrays['support']==255)
 report=dict(schema='renderello-refractive-transfer-result-v1',spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),input_read_semantics='bounded_immutable_byte_snapshots_hashed_before_decode',input_sha256=d['input_sha256'],map_origin=d['map_origin'],method='linear_sRGB_source_plus_transmission_times_warped_background_difference',quality='NOT_EVALUATED',real_glass_validated=False,limitations=LIMITATIONS,**stats)
 out.mkdir(parents=True);Image.fromarray(result).save(out/'candidate.png');report['output_sha256']={'candidate.png':sha(out/'candidate.png')};(out/'refractive-transfer-manifest.json').write_text(json.dumps(report,indent=2));return report
