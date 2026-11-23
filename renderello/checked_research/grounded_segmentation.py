"""Opt-in CPU box-to-mask research. Retains all three masks, never claims identities.

Native RGB is preserved in every RGBA export. Binary masks are not alpha matting.
Reflections, merged products, occlusion and missing bristles can survive this route.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .grounded_proposals import sha256, checked_proposals, MAX_PIXELS

MODEL = 'facebook/sam2.1-hiera-tiny'
REVISION = 'de431c4043854a71d8101e17995dfe596bf101a5'
WEIGHTS_SHA256 = '48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7'


def checked_manifest(value, source_hash, native_size):
    if not isinstance(value, dict) or value.get('source_sha256') != source_hash or value.get('native_size') != list(native_size):
        raise ValueError('proposal source hash or native dimensions differ')
    if value.get('status') != 'PROPOSALS_NOT_SEGMENTATIONS' or value.get('source_resampled') is not False:
        raise ValueError('native checked proposal manifest required')
    props = value.get('proposals')
    if not isinstance(props, list) or not 1 <= len(props) <= 64:
        raise ValueError('1..64 explicit proposals required; no silent limiting')
    result = []
    for i, p in enumerate(props):
        if not isinstance(p, dict) or type(p.get('proposal_id')) is not int or p['proposal_id'] != i:
            raise ValueError('stable consecutive integer proposal IDs required')
        try:
            one = checked_proposals([dict(query=p['query'], boxes=[p['box']], scores=[p['model_confidence']], labels=[p['predicted_text']])], *native_size)[0]
        except (KeyError, TypeError) as e:
            raise ValueError('proposal fields missing or malformed') from e
        one['proposal_id'] = i
        result.append(one)
    return result


def checked_masks(masks, scores, native_size):
    a, s = np.asarray(masks), np.asarray(scores)
    w, h = native_size
    if a.shape != (3, h, w) or a.dtype != np.bool_:
        raise ValueError('exactly three native-size boolean masks required; no resizing or threshold guessing')
    if s.shape != (3,) or (not np.issubdtype(s.dtype, np.floating) and not np.issubdtype(s.dtype, np.integer)) or not np.isfinite(s).all():
        raise ValueError('three finite model mask-quality scores required; not accuracy')
    area = a.sum(axis=(1,2))
    if np.any(area == 0) or np.any(area == w*h):
        raise ValueError('empty or all-image mask refused')
    return a, s.astype(np.float64)


class CpuSegmenter:
    def __init__(self, threads=2, cache_embeddings=True):
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU threads 1..4 required')
        import torch
        import transformers
        import torchvision
        from transformers import Sam2Model, Sam2Processor
        from huggingface_hub import hf_hub_download
        torch.set_num_threads(threads)
        self.cache_embeddings = cache_embeddings
        self.torch = torch
        weights = hf_hub_download(MODEL, 'model.safetensors', revision=REVISION)
        if sha256(weights) != WEIGHTS_SHA256:
            raise ValueError('pinned segmentation checkpoint hash differs')
        self.model, loading = Sam2Model.from_pretrained(MODEL, revision=REVISION, output_loading_info=True, use_safetensors=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('strict segmentation checkpoint load failed')
        self.model.eval()
        self.processor = Sam2Processor.from_pretrained(MODEL, revision=REVISION)
        self.embeddings = None
        self.source_rgb_hash = None
        self.info = dict(model=MODEL, revision=REVISION, weights_sha256=WEIGHTS_SHA256, loading_info=loading,
                         device='cpu', threads=threads, torch_version=torch.__version__,
                         transformers_version=transformers.__version__, torchvision_version=torchvision.__version__,
                         internal_resize=list(self.processor.image_processor.size.values()))

    def __call__(self, image, proposal):
        rgb_hash = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        if self.cache_embeddings and self.source_rgb_hash is not None and rgb_hash != self.source_rgb_hash:
            raise ValueError('cannot reuse image embeddings for a different image')
        self.source_rgb_hash = rgb_hash
        inputs = self.processor(images=image, input_boxes=[[proposal['box']]], return_tensors='pt')
        cached = self.embeddings is not None
        if cached:
            del inputs['pixel_values']
            inputs['image_embeddings'] = self.embeddings
        start = time.monotonic()
        with self.torch.inference_mode():
            out = self.model(**inputs, multimask_output=True)
        seconds = time.monotonic() - start
        self.embeddings = out.image_embeddings if self.cache_embeddings else None
        masks = self.processor.post_process_masks(out.pred_masks, inputs['original_sizes'])[0][0].numpy()
        return dict(proposal_id=proposal['proposal_id'], masks=masks, scores=out.iou_scores[0,0].numpy(),
                    seconds=seconds, includes_image_embedding=not cached)


def native_crop(image, proposal, margin):
    """Explicit proposal-centred native crop; cannot recover missing proposal objects."""
    if type(margin) is not int or not 1 <= margin <= 256:
        raise ValueError('explicit crop margin must be integer 1..256')
    x0,y0,x1,y1 = proposal['box']
    if not np.isfinite([x0,y0,x1,y1]).all() or not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError('native proposal bounds required')
    bounds = (max(0,int(np.floor(x0))-margin), max(0,int(np.floor(y0))-margin),
              min(image.width,int(np.ceil(x1))+margin), min(image.height,int(np.ceil(y1))+margin))
    local = dict(proposal, box=[x0-bounds[0],y0-bounds[1],x1-bounds[0],y1-bounds[1]])
    return image.crop(bounds), local, bounds


def restore_native_masks(masks, scores, crop, native_size):
    left,top,right,bottom = crop
    w,h = native_size
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError('crop must lie inside native source')
    small, scores = checked_masks(masks, scores, (right-left,bottom-top))
    native = np.zeros((3,h,w),dtype=bool)
    native[:,top:bottom,left:right] = small
    return native, scores


def run(source, proposals_path, output_dir, *, source_sha256, proposals_sha256, threads=2, segmenter=None, crop_margin=None):
    if crop_margin is not None and (type(crop_margin) is not int or not 1 <= crop_margin <= 256):
        raise ValueError('explicit crop margin must be integer 1..256')
    if sha256(source) != source_sha256 or sha256(proposals_path) != proposals_sha256:
        raise ValueError('source or proposal-file hash differs')
    dest = Path(output_dir)
    if dest.exists():
        raise ValueError('output directory exists; no replacement')
    with Image.open(source) as im:
        if im.mode != 'RGB' or min(im.size) < 2 or im.width*im.height > MAX_PIXELS:
            raise ValueError('native RGB image within 2M-pixel CPU bound required')
        image = im.copy()
    props = checked_manifest(json.loads(Path(proposals_path).read_text()), source_sha256, image.size)
    worker = segmenter if segmenter is not None else CpuSegmenter(threads,cache_embeddings=crop_margin is None)
    results = []
    for p in props:
        crop = None
        if crop_margin is None:
            row = worker(image, p)
        else:
            local_image, local_proposal, crop = native_crop(image,p,crop_margin)
            row = worker(local_image,local_proposal)
        if type(row.get('proposal_id')) is not int or row['proposal_id'] != p['proposal_id']:
            raise ValueError('segmenter changed proposal provenance')
        masks, scores = checked_masks(row['masks'], row['scores'], image.size) if crop is None else restore_native_masks(row['masks'],row['scores'],crop,image.size)
        row = dict(row, native_crop_bounds=crop)
        seconds = row.get('seconds')
        if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError('finite nonnegative inference timing required')
        if type(row.get('includes_image_embedding')) is not bool:
            raise ValueError('explicit image-embedding timing provenance required')
        results.append((p, row, masks, scores))
    dest.mkdir(parents=True)
    image.save(dest/'source-native.png')
    rgb = np.asarray(image)
    records = []
    for p, row, masks, scores in results:
        i = p['proposal_id']
        np.savez_compressed(dest/f'proposal-{i:02d}-raw.npz', masks=masks, model_mask_quality=scores)
        names = []
        for j, mask in enumerate(masks):
            name = f'proposal-{i:02d}-candidate-{j}.png'
            Image.fromarray(np.dstack((rgb, mask.astype(np.uint8)*255))).save(dest/name)
            names.append(name)
        records.append(dict(proposal=p, candidates=names, model_mask_quality=scores.tolist(),
                            areas=masks.sum(axis=(1,2)).tolist(), seconds=row['seconds'],
                            includes_image_embedding=row['includes_image_embedding'], native_crop_bounds=row['native_crop_bounds'], selected_candidate=None,
                            status='MASK_CANDIDATES_NOT_VERIFIED_PRODUCTS'))
    info = dict(source_sha256=source_sha256, proposals_sha256=proposals_sha256, native_size=list(image.size),
                crop_margin=crop_margin, crop_limits_context=crop_margin is not None, source_resampled=False, source_rgb_preserved=True, binary_masks_not_matting=True,
                segmenter=getattr(worker,'info',{'unverified_injected_segmenter':True}), records=records,
                production_default_changed=False, publication_accepted=False, accuracy=None, completeness=None)
    (dest/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-sha256','proposals-path','proposals-sha256','output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--crop-margin',type=int,default=None,help='Opt-in native crop padding 1..256; can lose image context')
    args = parser.parse_args()
    try:
        result = run(args.source,args.proposals_path,args.output_dir,source_sha256=args.source_sha256,
                     proposals_sha256=args.proposals_sha256,threads=args.threads,crop_margin=args.crop_margin)
    except (ValueError,OSError) as e:
        parser.exit(2,'REFUSED: '+str(e)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
