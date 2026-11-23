"""Build a sealed blank blind pack from an explicit local JSON manifest."""
import argparse,json
from pathlib import Path
from .blind_review import build_pack,Candidate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.manifest.is_symlink() or not a.manifest.is_file() or a.manifest.stat().st_size>20000:raise ValueError('bounded manifest required')
        m=json.loads(a.manifest.read_text())
        if not isinstance(m,dict) or set(m)-{'source','seed','candidates','regions'}:raise ValueError('exact pack manifest fields required')
        root=a.manifest.resolve().parent
        def path(v):return root/Path(v)
        cs=[]
        for c in m['candidates']:
            if not isinstance(c,dict) or set(c)-{'method','path','failure'}:raise ValueError('unknown candidate fields')
            cs.append(Candidate(c['method'],None if c.get('path') is None else path(c['path']),c.get('failure','')))
        out=build_pack(path(m['source']),cs,a.out,seed=m['seed'],regions=m.get('regions'))
    except (ValueError,TypeError,KeyError,OSError) as exc:p.exit(2,'REFUSED: '+str(exc)+'\n')
    print('Saved sealed blank pack: '+str(out)+'; share REVIEW only, never PRIVATE. No scores inferred.')


if __name__=='__main__':main()
