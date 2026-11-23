"""Recompute opaque RGB audit from recorded placement and a separately regenerated mask.
The supplied alpha is the locally reproduced pinned model mask, not a download of
production's intermediate. This audit does not validate edge quality or lighting.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from renderello.native_pixel_audit import audit_native_pixels
from renderello.matte_integrity import matte_stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['source','alpha','output','manifest','report']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();m=json.loads(a.manifest.read_text());entry=m['variants'][0]['qa']['native_pixel_audit'];raw=a.source.read_bytes()
 if hashlib.sha256(raw).hexdigest()[:16]!=m['products'][0]['sha16']:raise ValueError('source digest mismatch')
 if m['provider_used']!='magichour' or m['model']!='seedream-v5-flash' or not m['live']:raise ValueError('wrong provider evidence')
 s=np.array(Image.open(a.source).convert('RGB'));alpha=np.array(Image.open(a.alpha));out=np.array(Image.open(a.output).convert('RGB'))
 if matte_stats(alpha)!=m['products'][0]['matte_integrity']:raise ValueError('locally reproduced matte stats differ')
 result=audit_native_pixels(s,alpha,out,entry['source_box'],entry['output_box'])
 result.update(alpha_basis='LOCAL_REPRODUCTION_OF_PINNED_DEPLOYED_BACKEND; production intermediate not downloaded',source_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),live_audit_agrees=result==entry,human_review='PENDING')
 a.report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if result['status']!='MATCH':raise SystemExit(1)
if __name__=='__main__':main()
