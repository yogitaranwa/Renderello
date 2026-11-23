"""Local anonymized review packs. No ratings or human identity are inferred.

Candidate order is seeded for reproducibility, not cryptographic secrecy. The
private key must be kept away from raters. Missing/failed outputs remain entries.
Full-resolution PNG derivatives accompany scaled overviews for edge/print review.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import random
from PIL import Image, ImageDraw
from .image_intake import inspect_image, IntakePolicy
from .review import DIMENSIONS


@dataclass(frozen=True)
class Candidate:
    method: str
    path: Path | None
    failure: str = ''

    def __post_init__(self):
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError('Method identity required in private key')
        if not isinstance(self.failure, str):
            raise ValueError('Failure must be text')
        if self.path is None and not self.failure.strip():
            raise ValueError('Missing output requires an explicit failure')
        if self.path is not None and self.failure:
            raise ValueError('Candidate cannot both succeed and fail')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pack(source: Path, candidates: list[Candidate], out: Path, *, seed: int,
               regions: dict[str, tuple[int, int, int, int]] | None = None) -> Path:
    """Build one source comparison; invalid input is not silently skipped.

    Regions use the shared full-image coordinate system. Candidates with different
    dimensions are refused rather than making an ungrounded registration/print
    fidelity claim. More general layouts need their own declared transforms.
    """
    if not 2 <= len(candidates) <= 12:
        raise ValueError('Review pack requires 2-12 candidate entries')
    if len({c.method for c in candidates}) != len(candidates):
        raise ValueError('Duplicate method identity')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Integer seed required')
    if out.exists():
        raise FileExistsError('Never replace an earlier review')
    source = Path(source)
    original_hash = _sha(source)
    policy = IntakePolicy(allow_colour_conversion=True)
    src = inspect_image(source, policy)
    order = list(candidates)
    random.Random(seed).shuffle(order)
    decoded = []
    seen = set()
    decoded_seen = set()
    try:
        for c in order:
            if c.path is None:
                decoded.append((c, None, None));continue
            path = Path(c.path)
            digest = _sha(path)
            if digest in seen:
                raise ValueError('Duplicate output content cannot inflate the comparison')
            seen.add(digest)
            image = inspect_image(path, policy)
            if image.image.size != src.image.size:
                image.close()
                raise ValueError('Candidate/source dimensions differ; registration is not established')
            canonical = image.image.convert('RGBA')
            try:
                pixel_digest = hashlib.sha256(canonical.tobytes()).hexdigest()
            finally:
                canonical.close()
            if pixel_digest in decoded_seen:
                image.close()
                raise ValueError('Duplicate output pixels cannot inflate the comparison')
            decoded_seen.add(pixel_digest)
            decoded.append((c, image, digest))
        regions = regions or {}
        for label, box in regions.items():
            if not label.strip() or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in box):
                raise ValueError('Named integer ROI required')
            x0,y0,x1,y1=box;w,h=src.image.size
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError('ROI outside shared image coordinates')
        if _sha(source) != original_hash:
            raise ValueError('Source changed during review preparation')
        for c,im,digest in decoded:
            if im is not None and _sha(c.path) != digest:
                raise ValueError('Candidate changed during review preparation')
        out.mkdir(parents=True)
        # A pack is consumable only after COMPLETE.json exists and hashes verify.
        (out/'PRIVATE').mkdir();(out/'REVIEW').mkdir()
        src.image.save(out/'REVIEW/source.png')
        rows=[];key=[];tiles=[('SOURCE',src.image)]
        for i,(c,im,digest) in enumerate(decoded,1):
            token='C'+str(i).zfill(2)
            key.append(dict(token=token,method=c.method,path=str(c.path) if c.path else None,
                            original_sha256=digest,failure=c.failure))
            if im is not None:
                im.image.save(out/'REVIEW'/f'{token}.png')
                for j,box in enumerate(regions.values(),1):
                    im.image.crop(box).save(out/'REVIEW'/f'{token}-region{j}.png')
            tiles.append((token,im.image if im else None))
            rows.append(dict(token=token,status='OUTPUT_PRESENT' if im else 'NOT_GENERATED',
                             **{name:'' for name in DIMENSIONS},no_critical='',critical_errors='',rater='',notes=''))
        for j,box in enumerate(regions.values(),1):
            src.image.crop(box).save(out/'REVIEW'/f'source-region{j}.png')
        board=Image.new('RGB',(1200,430*((len(tiles)+2)//3)+40),'white');draw=ImageDraw.Draw(board)
        draw.text((12,10),'Scaled overview only. Inspect full-resolution images and native crops before rating.',fill='black')
        for i,(token,im) in enumerate(tiles):
            x=(i%3)*400;y=40+(i//3)*430;draw.text((x+10,y),token,fill='black')
            if im is None:
                draw.text((x+10,y+50),'NOT GENERATED / retained in denominator',fill='red');continue
            # Checkerboard makes partial alpha visible, not a fake opaque white pass.
            checker=Image.new('RGB',(380,390),'#ddd');cd=ImageDraw.Draw(checker)
            for yy in range(0,390,20):
                for xx in range(0,380,20):
                    if (xx//20+yy//20)%2:cd.rectangle((xx,yy,xx+19,yy+19),fill='#aaa')
            preview=im.copy();preview.thumbnail((380,390))
            px=(380-preview.width)//2;py=(390-preview.height)//2
            checker.paste(preview,(px,py),preview.getchannel('A') if preview.mode=='RGBA' else None)
            board.paste(checker,(x+10,y+25));preview.close()
        board.save(out/'REVIEW/overview.png');board.close()
        fields=list(rows[0])
        with (out/'REVIEW/scores.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        (out/'PRIVATE/key.json').write_text(json.dumps(dict(seed=seed,source_path=str(source),source_sha256=original_hash,candidates=key,regions=regions),indent=2))
        notes=('Share only REVIEW, never PRIVATE, with raters. Token assignment is randomized but not proof of independent blinding. '
               'No automatic score, winner or acceptance is inferred. Missing outputs stay in the denominator. '
               'ICC conversion may alter colours; full-resolution images are decoded normalized derivatives, not original encoded bytes. '
               'Independent rater identity must be confirmed outside this file.\n')
        (out/'REVIEW/READ_ME.txt').write_text(notes)
        hashes={str(p.relative_to(out)):_sha(p) for p in (out/'REVIEW').iterdir() if p.is_file()}
        (out/'COMPLETE.json').write_text(json.dumps(dict(schema_version=2,private_key_sha256=_sha(out/'PRIVATE/key.json'),candidates=len(candidates),outputs=sum(im is not None for _,im,_ in decoded),ratings='BLANK',acceptance='NOT_EVALUATED',review_sha256=hashes),indent=2))
        return out
    finally:
        src.close()
        for _,im,_ in decoded:
            if im is not None:im.close()


def verify_pack(out: Path) -> dict:
    """Reject incomplete or altered packs without accepting any ratings."""
    record=json.loads((out/'COMPLETE.json').read_text())
    if record.get('schema_version') == 2:
        key_path = out/'PRIVATE/key.json'
        if key_path.is_symlink() or key_path.resolve().parent != (out/'PRIVATE').resolve() or _sha(key_path) != record.get('private_key_sha256'):
            raise ValueError('Private candidate key changed or is missing')
    expected=record['review_sha256']
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Empty or invalid review manifest')
    review = (out/'REVIEW').resolve()
    for name, digest in expected.items():
        path = out / name
        if path.is_symlink() or path.resolve().parent != review:
            raise ValueError('Review manifest path escapes review directory')
        if _sha(path) != digest:
            raise ValueError('Review content changed or is missing')
    actual = {str(p.relative_to(out)) for p in (out/'REVIEW').iterdir()}
    if actual != set(expected):
        raise ValueError('Unexpected review files')
    return record
