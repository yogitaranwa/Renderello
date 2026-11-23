"""Explicit lossless RGB-container lineage, never rights or photograph clearance.

Not an upload-name heuristic or default match. Both bounded original and derived
files are supplied by the caller. A report describes these snapshots only.
"""
from io import BytesIO
from pathlib import Path
import hashlib,json,os,stat
import numpy as np
from PIL import Image

MAX_BYTES=16_000_000
MAX_PIXELS=1_000_000

def _snapshot(path):
    p=Path(path)
    if p.is_symlink():raise ValueError('Regular bounded source file required')
    with os.fdopen(os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)),'rb') as f:
        st=os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size>MAX_BYTES:raise ValueError('Regular bounded source file required')
        data=f.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:raise ValueError('Source byte limit exceeded')
    return data

def _rgb(data):
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        pos=8
        while pos<len(data):
            if len(data)-pos<12:raise ValueError('Incomplete PNG metadata stream')
            length=int.from_bytes(data[pos:pos+4],'big');kind=data[pos+4:pos+8]
            if pos+12+length>len(data):raise ValueError('Incomplete PNG metadata stream')
            if kind in {b'cICP',b'mDCV',b'cLLI',b'tRNS',b'acTL',b'fcTL',b'fdAT'}:
                raise ValueError('Unsupported colour, transparency, or animation metadata')
            pos+=12+length
    with Image.open(BytesIO(data)) as im:
        if im.format not in ('JPEG','PNG') or im.mode!='RGB':raise ValueError('Native RGB JPEG or PNG required; no conversion')
        if im.width*im.height>MAX_PIXELS or getattr(im,'n_frames',1)!=1:raise ValueError('Single RGB frame up to one megapixel required')
        if im.getexif().get(274,1)!=1:raise ValueError('Orientation transform not permitted')
        # Even visually equivalent metadata must not be silently discarded.
        color={k:im.info.get(k) for k in ('icc_profile','gamma','srgb','chromaticity')}
        color['exif']=dict(im.getexif())
        return np.array(im),color,im.format

def verify_lossless_container_lineage(original,derived):
    """Return explicit content ancestry only; leave default source lookup alone."""
    a,b=_snapshot(original),_snapshot(derived)
    rgb_a,color_a,fmt_a=_rgb(a);rgb_b,color_b,fmt_b=_rgb(b)
    if fmt_b!='PNG':raise ValueError('Derived container must be PNG')
    if rgb_a.shape!=rgb_b.shape or not np.array_equal(rgb_a,rgb_b):raise ValueError('Decoded native RGB changed')
    if color_a!=color_b:raise ValueError('Colour or EXIF interpretation metadata changed')
    ha,hb=(hashlib.sha256(v).hexdigest() for v in (a,b))
    registry=Path(__file__).with_name('source-registry.json')
    records=json.loads(registry.read_text()) if registry.is_file() else []
    matches=[r for r in records if r.get('sha256')==ha]
    if len(matches)>1:raise ValueError('Ambiguous parent source registry')
    return dict(schema='renderello-explicit-source-lineage.v1',original_file_sha256=ha,
        derived_file_sha256=hb,original_container=fmt_a,derived_container=fmt_b,
        size=[rgb_a.shape[1],rgb_a.shape[0]],decoded_rgb_exact=True,
        interpretation_metadata_exact=True,parent_record=matches[0] if matches else None,
        parent_record_status='Historical content-hash lookup only; not authenticated authorship or clearance',
        source_photo_origin_verified=False,rights_verified=False,publication_approved=False,
        default_source_lookup_changed=False,
        caveat='Caller-supplied ancestry verified for these file snapshots only. RGB equality does not prove genuine capture, author identity, licensing, alpha accuracy, or scene quality.')
