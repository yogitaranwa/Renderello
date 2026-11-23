"""Opt-in text-grounded native-coordinate box proposals, not product masks.

The detector resizes its internal input explicitly. Saved source pixels and output
coordinates remain source-native. Scores are not accuracy; reflected objects and
wrong categories can remain. This route never selects the largest object.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

MODEL = 'IDEA-Research/grounding-dino-tiny'
REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
WEIGHTS_SHA256 = '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
MAX_PIXELS = 2_000_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_queries(queries):
    if not isinstance(queries, list) or not 1 <= len(queries) <= 12:
        raise ValueError('explicit list of 1..12 category queries required')
    for q in queries:
        if not isinstance(q, str) or not q.strip() or q != q.strip() or len(q) > 256 or not q.endswith('.') or q != q.lower():
            raise ValueError('lowercase query ending in period, at most 256 characters required')
    if len(set(queries)) != len(queries):
        raise ValueError('duplicate query')
    return queries


def checked_proposals(rows, width, height):
    """Retain all predictions and their query provenance; never infer identity."""
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('native integer dimensions required')
    result = []
    for row in rows:
        if not isinstance(row.get('query'), str) or not row['query']:
            raise ValueError('query provenance required')
        boxes, scores, labels = row['boxes'], row['scores'], row['labels']
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError('box/score/label count differs')
        for box, score, label in zip(boxes, scores, labels):
            b = np.asarray(box, dtype=np.float64)
            if b.shape != (4,) or not np.isfinite(b).all() or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('finite box and score in [0,1] required')
            if not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height):
                raise ValueError('source-native in-bounds nondegenerate box required; no clipping')
            if not isinstance(label, str) or not label:
                raise ValueError('nonempty predicted text required')
            result.append(dict(proposal_id=len(result), query=row['query'], box=b.tolist(),
                               model_confidence=float(score), predicted_text=label))
    return result


def group_overlaps(proposals, threshold=0.75):
    """Greedy box overlap groups, NOT deduplicated product identities.

    Every member survives, including incompatible category predictions. Grouping
    order is stable confidence-descending, then input order. A group is matched
    to its representative, not transitive-connected to all its other members.
    """
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('overlap IoU threshold in (0,1] required')
    groups = []
    for p in sorted(proposals, key=lambda x: (-x['model_confidence'], x['proposal_id'])):
        a = p['box']
        for g in groups:
            b = g['representative']['box']
            inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            if inter / union >= threshold:
                g['members'].append(p)
                break
        else:
            groups.append(dict(group_id=len(groups), representative=p, members=[p]))
    return groups


class CpuPredictor:
    def __init__(self, threads=2):
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU threads 1..4 required')
        import torch
        import transformers
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
        from huggingface_hub import hf_hub_download
        torch.set_num_threads(threads)
        self.torch = torch
        weights = hf_hub_download(MODEL, 'model.safetensors', revision=REVISION)
        if sha256(weights) != WEIGHTS_SHA256:
            raise ValueError('pinned checkpoint hash differs')
        self.processor = GroundingDinoProcessor.from_pretrained(MODEL, revision=REVISION)
        self.processor.image_processor.size = {'shortest_edge':480, 'longest_edge':800}
        self.model, loading = GroundingDinoForObjectDetection.from_pretrained(
            MODEL, revision=REVISION, use_safetensors=True, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('strict checkpoint load failed')
        self.model.eval()
        self.info = dict(model=MODEL, revision=REVISION, weights_sha256=WEIGHTS_SHA256,
                         loading_info=loading, device='cpu', threads=threads,
                         torch_version=torch.__version__, transformers_version=transformers.__version__,
                         detector_internal_resize={'shortest_edge':480,'longest_edge':800})

    def __call__(self, image, query):
        tokens = self.processor.tokenizer(query, truncation=False)['input_ids']
        if len(tokens) > self.model.config.max_text_len:
            raise ValueError('query exceeds checkpoint token limit; no silent truncation')
        inputs = self.processor(images=image, text=query, return_tensors='pt')
        if inputs.input_ids.shape[-1] != len(tokens):
            raise ValueError('processor changed query token length')
        t = time.monotonic()
        with self.torch.inference_mode():
            out = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=.25, text_threshold=.25,
            target_sizes=[image.size[::-1]])[0]
        return dict(query=query, boxes=result['boxes'].tolist(), scores=result['scores'].tolist(),
                    labels=result['text_labels'], input_shape=list(inputs.pixel_values.shape),
                    seconds=time.monotonic()-t)


def run(source, output_dir, *, source_sha256, queries, threads=2, predictor=None):
    queries = validate_queries(queries)
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(x not in '0123456789abcdef' for x in source_sha256):
        raise ValueError('explicit lowercase SHA-256 required')
    if sha256(source) != source_sha256:
        raise ValueError('source hash differs')
    dest = Path(output_dir)
    if dest.exists():
        raise ValueError('output directory exists; no replacement')
    with Image.open(source) as im:
        if im.mode != 'RGB' or min(im.size) < 2 or im.width*im.height > MAX_PIXELS:
            raise ValueError('native RGB source within 2M pixel CPU bound required')
        image = im.copy()
    worker = predictor if predictor is not None else CpuPredictor(threads)
    rows = [worker(image, q) for q in queries]
    for expected, row in zip(queries, rows):
        if row.get('query') != expected:
            raise ValueError('predictor changed query provenance')
    props = checked_proposals(rows, *image.size)
    manifest = dict(source_sha256=source_sha256, native_size=list(image.size), source_resampled=False,
                    predictor=getattr(worker,'info',{'unverified_injected_predictor':True}),
                    box_threshold=.25,text_threshold=.25,queries=rows,proposals=props,
                    overlap_groups=group_overlaps(props),grouping_iou=.75,
                    status='PROPOSALS_NOT_SEGMENTATIONS',accuracy=None,completeness=None,
                    publication_accepted=False,production_default_changed=False)
    dest.mkdir(parents=True)  # exclusive, only after complete validated inference
    image.save(dest/'source-native.png')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','source-sha256','queries-json','output-dir'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    try:
        value=run(a.source,a.output_dir,source_sha256=a.source_sha256,
                  queries=json.loads(a.queries_json),threads=a.threads)
    except (ValueError,OSError) as e:
        p.exit(2,'REFUSED: '+str(e)+'\n')
    print(json.dumps(value,indent=2))

if __name__=='__main__':
    main()
