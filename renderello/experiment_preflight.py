"""Read-only preflight for local cases x seeds image experiments.

Validates declared execution metadata, never source rights, quality, permission,
provider cost or successful execution. Frozen evidence is never repaired here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
from .matched_trial import read_json, _relative


def _int(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label + ': bounded integer required')
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 32000:
        raise ValueError(label + ': nonempty bounded text required')
    return value


def _pair(value, label, low, high):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(label + ': two integers required')
    return [_int(x, label, low, high) for x in value]


def validate_plan(plan, root):
    """Derive output count and audit source bytes/canvases, with no mutations.

    Source-shaped canvases are allowed, but a conflicting template aspect is
    reported as unmet, never silently counted as fulfilled. Multiple template
    cases can share one source. Extra provenance rows are not executed sources.
    """
    if not isinstance(plan, dict):
        raise ValueError('plan object required')
    cases = plan.get('cases'); seeds = plan.get('seeds'); sources = plan.get('sources')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 500:
        raise ValueError('one to 500 cases required')
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 16:
        raise ValueError('one to 16 seeds required')
    for s in seeds: _int(s, 'seed', 0, 2**63-1)
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate seeds')
    count = len(cases)*len(seeds)
    if _int(plan.get('planned_outputs'), 'planned_outputs', 1, 8000) != count:
        raise ValueError(f'planned_outputs contradicts cases x seeds: expected {count}')
    if not isinstance(sources, list) or not 1 <= len(sources) <= 500:
        raise ValueError('source records required')
    records = {}
    for row in sources:
        if not isinstance(row, dict): raise ValueError('source object required')
        sid = _text(row.get('id'), 'source id')
        if sid in records: raise ValueError('duplicate source id')
        records[sid] = row
    rows=[];seen=set();used=set();cells=set()
    root=Path(root)
    for c in cases:
        if not isinstance(c,dict): raise ValueError('case object required')
        cid=_text(c.get('id'),'case id');sid=_text(c.get('source_id'),'case source id')
        if cid in seen: raise ValueError('duplicate case id')
        seen.add(cid);used.add(sid)
        if sid not in records: raise ValueError('case has no source record')
        record=records[sid]
        if c.get('source') != record.get('input_file'):
            raise ValueError('case/source record path mismatch')
        path=_relative(root,c.get('source'))
        if path.stat().st_size>64_000_000: raise ValueError('source file exceeds limit')
        source_bytes=path.read_bytes()
        if len(source_bytes)>64_000_000: raise ValueError('source file exceeds limit')
        digest=hashlib.sha256(source_bytes).hexdigest()
        if digest != record.get('input_file_sha256'):
            raise ValueError('source bytes do not match input_file_sha256')
        with Image.open(BytesIO(source_bytes)) as im:
            if im.format not in {'PNG','JPEG','WEBP'} or getattr(im,'n_frames',1)!=1:
                raise ValueError('single-frame raster source required')
            size=list(im.size)
            if size[0]*size[1]>40_000_000: raise ValueError('source pixel limit exceeded')
            if im.getexif().get(274,1)!=1: raise ValueError('normalize orientation before freezing')
            im.load()
        with Image.open(BytesIO(source_bytes)) as check:
            check.verify()
        if size != _pair(record.get('decoded_size'),'decoded size',1,16384):
            raise ValueError('decoded source size mismatch')
        width=_int(c.get('width'),'width',32,8192);height=_int(c.get('height'),'height',32,8192)
        if width*height>40_000_000 or width%32 or height%32:
            raise ValueError('bounded multiple-of-32 canvas required')
        if [width,height] != _pair(record.get('output_canvas'),'output canvas',32,8192):
            raise ValueError('source/case canvas mismatch')
        xy=_pair(c.get('translation'),'translation',0,8192)
        if xy != _pair(record.get('native_translation'),'native translation',0,8192):
            raise ValueError('source/case translation mismatch')
        if record.get('source_resampled') is not False or record.get('canvas_only_padding') is not True:
            raise ValueError('explicit native-padding declarations required')
        if xy[0]+size[0]>width or xy[1]+size[1]>height:
            raise ValueError('native source extends outside canvas')
        prompt=_text(c.get('prompt'),'prompt');template=c.get('template')
        if not isinstance(template,dict): raise ValueError('template object required')
        key=_text(template.get('key'),'template key');aspect=template.get('aspect')
        if not isinstance(aspect,str) or aspect.count(':')!=1:
            raise ValueError('integer template aspect required')
        try: aw,ah=(int(x) for x in aspect.split(':'))
        except (TypeError,ValueError): raise ValueError('integer template aspect required') from None
        if not 1<=aw<=100 or not 1<=ah<=100: raise ValueError('bounded positive aspect required')
        cell=(sid,key,prompt,width,height)
        if cell in cells: raise ValueError('duplicate logical case')
        cells.add(cell)
        rows.append(dict(case_id=cid,source_id=sid,input_sha256=digest,source_size=size,
                         canvas=[width,height],source_resampled=False,
                         requested_aspect=aspect,requested_aspect_fulfilled=width*ah==height*aw))
    return dict(schema='renderello.experiment-preflight.v1',validation='PASSED',
                planned_outputs=count,case_count=len(cases),seed_count=len(seeds),
                executed_source_ids=sorted(used),unused_provenance_source_ids=sorted(set(records)-used),
                case_checks=rows,all_template_aspects_fulfilled=all(r['requested_aspect_fulfilled'] for r in rows),
                generation_executed=False,quality_accepted=False,source_rights_verified=False,
                execution_authorized=False,cost_verified=False,files_modified=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan');p.add_argument('--root',required=True)
    a=p.parse_args(argv)
    try: result=validate_plan(read_json(a.plan),a.root)
    except (ValueError,KeyError,OSError,Image.DecompressionBombError) as exc:
        print(json.dumps(dict(validation='REJECTED',error=str(exc),generation_executed=False,files_modified=False)))
        return 2
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__': raise SystemExit(main())
