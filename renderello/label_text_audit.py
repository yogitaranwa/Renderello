"""Optional local OCR mismatch tripwire, never exact-text/brand acceptance.

Uses an installed Tesseract binary only. No cloud upload, download or default
pipeline change. OCR can misread legitimate print, miss invented print, and
ignore small text. A quiet result never certifies identity or publication.
"""
from __future__ import annotations
from collections import Counter
import csv, io, re, shutil, subprocess
from PIL import Image


def parse_tsv(text: str, *, confidence: float=70., min_length: int=4) -> list[dict]:
    if not isinstance(text,str):raise ValueError('TSV text required')
    if type(confidence) not in (int,float) or not 0<=confidence<=100:raise ValueError('confidence0..100 required')
    if type(min_length) is not int or not 1<=min_length<=40:raise ValueError('min_length1..40 required')
    if len(text)>8_000_000:raise ValueError('OCR report exceeds8MB budget')
    rows=csv.DictReader(io.StringIO(text),delimiter='\t')
    required={'level','conf','text','left','top','width','height'}
    if not required.issubset(rows.fieldnames or []):raise ValueError('Unexpected Tesseract TSV header')
    out=[]
    for row in rows:
        if row.get('level')!='5':continue
        try:c=float(row['conf']);box=[int(row[k])for k in ('left','top','width','height')]
        except (ValueError,TypeError):continue
        if not confidence<=c<=100 or min(box)<0 or min(box[2:])<=0:continue
        token=re.sub('[^A-Z0-9]','',(row.get('text')or'').upper())
        if len(token)<min_length:continue
        out.append(dict(token=token,observed_text=row['text'],confidence=c,box=box))
    return out


def extract_words(image: Image.Image, *, confidence: float=70., timeout: int=30) -> list[dict]:
    if not isinstance(image,Image.Image)or image.mode!='RGB':raise ValueError('PIL RGB image required; no hidden colour conversion')
    if image.width*image.height>4_000_000:raise ValueError('OCR native4MPbudget; no silent downscale')
    if type(timeout)is not int or not 1<=timeout<=60:raise ValueError('timeout1..60 required')
    binary=shutil.which('tesseract')
    if not binary:raise RuntimeError('Optional Tesseract is not installed; no OCR result')
    b=io.BytesIO();image.save(b,format='PNG')
    result=subprocess.run([binary,'stdin','stdout','-l','eng','--psm','11','tsv'],input=b.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
    if result.returncode:raise RuntimeError('Tesseract failed: '+result.stderr.decode('utf-8','replace')[:500])
    return parse_tsv(result.stdout.decode('utf-8','replace'),confidence=confidence)


def compare_words(source: list[dict], output: list[dict], *, warning_recall: float=.8) -> dict:
    if type(warning_recall)not in(int,float)or not 0<warning_recall<=1:raise ValueError('warning_recall must be0..1')
    def counts(rows):
        if not isinstance(rows,list)or any(not isinstance(r,dict)or not isinstance(r.get('token'),str)or not re.fullmatch('[A-Z0-9]+',r['token'])for r in rows):raise ValueError('typed normalized OCR word records required')
        return Counter(r['token']for r in rows)
    a,b=counts(source),counts(output);n=sum(a.values());same=sum((a&b).values());recall=same/n if n else None
    # Fewer than3sourcewords cannot support a useful retention diagnostic.
    status='INSUFFICIENT_SOURCE_OCR'if n<3 else('OCR_MISMATCH_REVIEW_REQUIRED'if recall<warning_recall else 'NO_OCR_TRIPWIRE_NOT_A_TEXT_PASS')
    return dict(schema='renderello-label-ocr-audit.v1',status=status,source_recognized_words=n,output_recognized_words=sum(b.values()),matched_source_words=same,source_word_recall=recall,missing_source_words=list((a-b).elements()),new_output_words=list((b-a).elements()),source_words=source,output_words=output,product_text_verified=False,photo_accepted=False,default_pipeline_changed=False,limitations=['OCR is fallible and incomplete; exact normalized word overlap is not text-order, typography, logo or identity proof.','Lighting, resolution and viewpoint can change OCR without actual label changes.','High-confidence OCR may still misread print; manual native-pixel comparison remains necessary.'])


def audit_label_text(source: Image.Image, output: Image.Image) -> dict:
    return compare_words(extract_words(source),extract_words(output))
