"""
unpack_results.py
-----------------
Unpacks the result files of the paper (results/runs/*.json.gz) into runs/,
where the report commands of the runners read them:

    python unpack_results.py
    python c2kg_runner.py report --out ../runs/c2kg_L.json
    python wn18rr_runner.py report --out ../runs/wn_O.json

Existing files in runs/ are not overwritten unless --force is given.
"""

import argparse
import glob
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "results", "runs"))
DST = os.path.normpath(os.path.join(HERE, "..", "runs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.json.gz")))
    if not files:
        raise SystemExit(f"no result files in {SRC}")
    for f in files:
        out = os.path.join(DST, os.path.basename(f)[:-3])
        if os.path.exists(out) and not args.force:
            print(f"kept     {out} (exists; --force to overwrite)")
            continue
        with gzip.open(f, "rb") as a, open(out, "wb") as b:
            shutil.copyfileobj(a, b)
        print(f"unpacked {out}")


if __name__ == "__main__":
    main()
