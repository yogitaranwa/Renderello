"""Portable generated-scene evidence gallery, separate from benchmark acceptance.

Inputs are local review declarations, never authenticated user permission.
All referenced bytes are verified before output; scene attempts never pad the
product, benchmark, template or source-photo denominator. No model invocation.
"""
from pathlib import Path
import hashlib,html,json,re

DIMENSIONS=('product_label_fidelity','photorealism','shadow_contact_realism',
    'material_edges','composition','template_distinctness','publish_ready_usefulness')


def _text(value,name,limit=4000):
    if not isinstance(value,str)or not value.strip()or len(value)>limit:
        raise ValueError('bounded nonempty '+name+' required')
    return value


def _asset(root,entry):
    if not isinstance(entry,dict)or set(entry)!={'path','sha256'}:
        raise ValueError('asset path and exact SHA256 required')
    raw=entry['path'];h=entry['sha256']
    if not isinstance(raw,str)or not raw or '\\'in raw or Path(raw).is_absolute():
        raise ValueError('relative portable asset path required')
    if any(p in ('','.','..')for p in raw.split('/')):
        raise ValueError('non-traversing asset path required')
    path=(root/raw).resolve()
    if not path.is_relative_to(root):raise ValueError('asset escapes gallery root')
    if not isinstance(h,str)or not re.fullmatch('[a-f0-9]{64}',h):
        raise ValueError('encoded asset SHA256 required')
    if path.stat().st_size>20_000_000:raise ValueError('bounded20MBgalleryasset')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=h:raise ValueError('asset bytes changed')
    # No conversion, resizing or hidden image orientation normalization here.
    from PIL import Image
    from io import BytesIO
    with Image.open(BytesIO(data))as im:
        if im.format!='PNG'or im.mode not in ('RGB','RGBA')or im.width*im.height>4_000_000 or getattr(im,'n_frames',1)!=1 or im.getexif().get(274,1)!=1:
            raise ValueError('single-frame normalized boundedRGB/RGBA PNG required')
        im.load()
    return raw


def build_research_gallery(manifest_path,out_html):
    manifest_path=Path(manifest_path);root=manifest_path.parent.resolve();out_html=Path(out_html)
    if out_html.parent.resolve()!=root:raise ValueError('gallery must travel beside manifest')
    if out_html.exists():raise FileExistsError('preserve prior gallery')
    if manifest_path.stat().st_size>1_000_000:raise ValueError('bounded1MBmanifest')
    m=json.loads(manifest_path.read_text())
    if set(m)!={'schema','title','note','cases'}or m['schema']!='renderello-research-gallery.v1':
        raise ValueError('research gallery schema required')
    _text(m['title'],'title',200);_text(m['note'],'note',8000)
    if not isinstance(m['cases'],list)or not 1<=len(m['cases'])<=256:
        raise ValueError('1..256cases required')
    cards=[];seen=set();scene_hashes=set();product_hashes=set();sources=set();scene_ids=set();product_ids=set();scored=0;first_scene=0;first_product=0
    for case in m['cases']:
        allowed={'id','kind','attempt','intent','image','source','source_id','note','caption','scores','reviewer','critical_errors_status'}
        if not isinstance(case,dict)or set(case)!=allowed:raise ValueError('explicit case fields required')
        ident=case['id']
        if not isinstance(ident,str)or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,79}',ident)or ident in seen:
            raise ValueError('unique safe case id required')
        seen.add(ident)
        kind=case['kind'];attempt=case['attempt']
        if kind not in ('scene_only','product_composite')or attempt not in ('first_try','retry'):
            raise ValueError('case kind/attempt required')
        for key in ('intent','note','caption','critical_errors_status'):_text(case[key],key)
        image=_asset(root,case['image']);source=case['source'];score=case['scores']
        if kind=='scene_only':
            if source is not None or score is not None or case['source_id']is not None or case['reviewer']!='assistant_scene_inspection':
                raise ValueError('scene-only cannot claim product scores/source coverage')
            scene_ids.add(ident);scene_hashes.add(case['image']['sha256']);first_scene+=attempt=='first_try'
        else:
            _text(case['source_id'],'source identity',100);source_path=_asset(root,source)
            if case['reviewer']!='assistant_not_blind_or_independent':raise ValueError('explicit assistant scoring label required')
            if not isinstance(score,dict)or set(score)!=set(DIMENSIONS)or any(type(v)is not int or not 1<=v<=5 for v in score.values()):
                raise ValueError('all seven assistant dimensions require integer1..5scores')
            product_ids.add(ident);product_hashes.add(case['image']['sha256']);sources.add(source['sha256']);scored+=1;first_product+=attempt=='first_try'
        esc=html.escape
        pics=(f'<figure><img loading="lazy" src="{esc(source_path,quote=True)}" alt="Original source product"><figcaption>Source input; capture origin not verified</figcaption></figure>'if kind=='product_composite'else'')
        pics+=f'<figure><img loading="lazy" src="{esc(image,quote=True)}" alt="{esc(ident,quote=True)}"><figcaption>{esc(case["caption"])}</figcaption></figure>'
        scores=('<dl>'+''.join(f'<dt>{esc(k.replace("_"," "))}</dt><dd>{v}/5</dd>'for k,v in score.items())+'</dl>')if score is not None else'<p>No product score. Scene-only attempt.</p>'
        cards.append(f'<article id="{ident}"><h2>{esc(case["intent"])}</h2><p class="tag">{esc(kind)} | {esc(attempt)} | {esc(ident)}</p><p class="status">RESEARCH ONLY - NOT ACCEPTED</p><div class="pair">{pics}</div><p>{esc(case["note"])}</p>{scores}<p class="fine">Reviewer: {esc(case["reviewer"])}. Critical errors: {esc(case["critical_errors_status"])}</p></article>')
    counts=dict(scene_attempts=len(scene_ids),scene_distinct_encoded_artifacts=len(scene_hashes),first_try_scene_attempts=first_scene,
        product_composite_attempts=len(product_ids),product_distinct_encoded_artifacts=len(product_hashes),first_try_product_composite_attempts=first_product,
        distinct_source_encoded_artifacts=len(sources),assistant_seven_dimension_reviews=scored,
        benchmark_accepts=0,polished_template_accepts=0,benchmark_evaluated=False,
        caveat='Artifact hashes are byte identity, not visually distinct templates or verified photo capture. All gallery cases are research-only and NOT ACCEPTED. Counts are not full original benchmark coverage.')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+html.escape(m['title'])+'''</title><style>
*{box-sizing:border-box}body{max-width:1180px;margin:32px auto;padding:0 18px;font:16px/1.5 system-ui,sans-serif;color:#222;background:#f5f4ef}h1{font-size:30px;line-height:1.15}h2{font-size:21px}header,article{background:white;border:1px solid #d3d3cd;border-radius:12px;padding:20px;margin:22px 0}figure{margin:0;min-width:0}img{display:block;max-width:100%;width:100%;object-fit:contain;max-height:620px;background:#fafafa}figcaption,.fine{font-size:13px;color:#555;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:18px}.status{color:#8b3529;font-weight:650}.tag{font-size:13px;color:#555;overflow-wrap:anywhere}dl{display:grid;grid-template-columns:minmax(140px,1fr) 50px;max-width:420px;font-size:14px}dt,dd{margin:0;padding:3px 0}summary{cursor:pointer;padding:8px 0}.counts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}.counts div{padding:12px;background:#efeee8;border-radius:8px}.counts b{display:block;font-size:28px}nav{display:flex;flex-wrap:wrap;gap:12px}a{color:#174b7c}pre{font:13px/1.45 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:500px){h1{font-size:25px}header,article{padding:14px}body{margin:18px auto}}</style></head><body><header><h1>'''+html.escape(m['title'])+'</h1><p>'+html.escape(m['note'])+'</p><p class="status">No accepted template or product-photo benchmark result in this gallery.</p><p>Generated scenes and source-product composites are counted separately. Repeated attempts and failed prompts remain visible. Scores are assistant judgments, not blind or independent review.</p><div class="counts"><div><b>'+str(counts['scene_attempts'])+'</b> scene attempts</div><div><b>'+str(counts['product_composite_attempts'])+'</b> product composites</div><div><b>'+str(counts['distinct_source_encoded_artifacts'])+'</b> source files</div><div><b>0</b> accepted templates</div></div><details><summary>Exact counts and counting limits</summary><pre>'+html.escape(json.dumps(counts,indent=2))+'</pre></details><details><summary>Jump to an attempt</summary><nav>'+''.join(f'<a href="#{c["id"]}">{html.escape(c["id"])}</a>'for c in m['cases'])+'</nav></details></header>'+''.join(cards)+'</body></html>'
    # Exclusive create only after all validation and byte checks have passed.
    with out_html.open('x',encoding='utf-8')as f:f.write(doc)
    return counts


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('output')
    args=parser.parse_args();print(json.dumps(build_research_gallery(args.manifest,args.output),indent=2))
