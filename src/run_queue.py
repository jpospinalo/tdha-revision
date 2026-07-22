"""
Modernized run_queue.py

This version preserves legacy behaviour while adding support for the new
pipeline parameters.
"""
import argparse,itertools,subprocess,sys
from pathlib import Path

def expand(v):
    return v if v else [None]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--script",default="src/run_experiment.py")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--jobs",type=int,default=1)
    p.add_argument("--sites",nargs="*")
    p.add_argument("--roi-sets",nargs="*")
    p.add_argument("--models",nargs="*")
    p.add_argument("--window-seconds",type=float,nargs="*")
    p.add_argument("--step-seconds",type=float,nargs="*")
    p.add_argument("--overlap",type=float,nargs="*")
    p.add_argument("--representation",nargs="*",choices=["dynamic","static"])
    p.add_argument("--window-shape",nargs="*")
    p.add_argument("--gaussian-sigma",type=float,nargs="*")
    p.add_argument("--fisher-z",action="store_true")
    args,unknown=p.parse_known_args()

    combos=list(itertools.product(
        expand(args.sites),
        expand(args.roi_sets),
        expand(args.models),
        expand(args.representation),
        expand(args.window_seconds),
        expand(args.step_seconds),
        expand(args.overlap),
        expand(args.window_shape),
        expand(args.gaussian_sigma),
    ))
    print(f"Experiments: {len(combos)}")
    for c in combos:
        site,roi,model,rep,w,s,o,shape,sigma=c
        cmd=[args.python,args.script]
        if site: cmd+=["--site",str(site)]
        if roi: cmd+=["--roi-set",str(roi)]
        if model: cmd+=["--model",str(model)]
        if rep:
            cmd+=["--representation",rep]
            if rep=="static":
                w=s=o=None
        if w is not None: cmd+=["--window-seconds",str(w)]
        if s is not None: cmd+=["--step-seconds",str(s)]
        if o is not None: cmd+=["--overlap",str(o)]
        if shape: cmd+=["--window-shape",shape]
        if sigma is not None: cmd+=["--gaussian-sigma",str(sigma)]
        if args.fisher_z: cmd.append("--fisher-z")
        cmd.extend(unknown)
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd,check=False)

if __name__=="__main__":
    main()
