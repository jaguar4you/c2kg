"""
audit_sensitivity.py
--------------------
Sensitivity of the leakage audit (leakage_audit.py) to its detection
threshold and minimum support.  For every combination it lists the
structures detected from the training graph and the number of test triples
with a mirror, inverse or co-referent counterpart in training.

    python audit_sensitivity.py > ../results/audit_threshold_sensitivity.txt
"""

import os

from leakage_audit import (load, symmetry_rates, inverse_candidates,
                           coreferent_candidates)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FOLDERS = ("c2kg_N", "c2kg_L", "wn18rr_O", "wn18rr_L")
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
SUPPORTS = (5, 20, 50)


def audit(train, test, threshold, support):
    known = set(train)
    rates, by_rel = symmetry_rates(train, support)
    sym = [r for r, v in sorted(rates.items(), key=lambda x: -(x[1] or 0.0))
           if v is not None and v >= threshold]
    invs = inverse_candidates(by_rel, threshold, support)
    cor = coreferent_candidates(by_rel, threshold, support)
    inv_map, cor_map = {}, {}
    for (a, b) in invs:
        inv_map.setdefault(a, set()).add(b)
    for (a, b) in cor:
        cor_map.setdefault(a, set()).add(b)
        cor_map.setdefault(b, set()).add(a)
    n_any = sum(1 for (h, r, t) in test
                if (t, r, h) in known
                or any((t, r2, h) in known for r2 in inv_map.get(r, ()))
                or any((h, r2, t) in known for r2 in cor_map.get(r, ())))
    n_inv = len({frozenset(p) for p in invs})
    return sym, n_inv, [f"{a}=={b}" for (a, b) in cor], n_any


def main():
    print("Sensitivity of the audit to the detection threshold and minimum support")
    print("(structures detected from the training graph; ANY = test triples with a mirror,")
    print(" inverse or co-referent counterpart in training)")
    for name in FOLDERS:
        d = os.path.join(DATA, name)
        train = load(os.path.join(d, "train.txt"))
        test = load(os.path.join(d, "test.txt"))
        print(f"\n== {name}")
        for th in THRESHOLDS:
            for sup in SUPPORTS:
                sym, n_inv, cor, n_any = audit(train, test, th, sup)
                print(f"  threshold {th}  support {sup:3d} | symmetric {len(sym)}: "
                      f"{', '.join(sym) or '-'} | inverse pairs {n_inv} | "
                      f"co-referent {len(cor)}: {', '.join(cor) or '-'} | "
                      f"ANY {n_any} ({100 * n_any / len(test):.1f}%)")
    print()


if __name__ == "__main__":
    main()
