"""
check_data.py
-------------
Compares the fingerprint (data_sha, as stored in every results file) of
each data folder with the values recorded in ../data/FINGERPRINTS.json.

    python check_data.py              # check
    python check_data.py --write      # (maintainer) record current values
"""

import argparse
import json
import os

import c2kg_runner as C

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
MANIFEST = os.path.join(DATA, "FINGERPRINTS.json")


def folders():
    return sorted(d for d in os.listdir(DATA)
                  if os.path.exists(os.path.join(DATA, d, "train.txt"))
                  and os.path.exists(os.path.join(DATA, d, "entities.dict")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cur = {d: C.data_sha(os.path.join(DATA, d)) for d in folders()}
    if args.write:
        with open(MANIFEST, "w", newline="\n") as f:
            json.dump(cur, f, indent=2)
            f.write("\n")
        print(f"recorded {len(cur)} fingerprints in {MANIFEST}")
        return
    ref = json.load(open(MANIFEST))
    bad = 0
    for d in sorted(set(ref) | set(cur)):
        a, b = ref.get(d), cur.get(d)
        ok = a == b
        bad += not ok
        print(f"{d:18s} {(b or '-')[:12]:12s}  "
              f"{'OK' if ok else 'MISMATCH (expected ' + (a or '-')[:12] + ')'}")
    if bad:
        raise SystemExit(f"{bad} folder(s) differ from FINGERPRINTS.json")
    print("all data folders match")


if __name__ == "__main__":
    main()
