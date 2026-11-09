"""
make_blocked.py
---------------
Leak-blocked variant of a split: the same validation and test triples, and
a training set from which every triple that restates a test triple has been
removed.

For each test triple (h, r, t) the restating training triples are those the
leakage audit counts (leakage_audit.py, same detection settings):

  mirror       (t, r, h)
  inverse      (t, r2, h)  for every r2 detected as the inverse of r
  co-referent  (h, r2, t)  for every r2 detected as co-referent with r

They are removed from train.txt and written to filter_extra.txt, so that
they are still filtered at evaluation (they are true facts) but neither
trained on nor evaluated.  A partner is kept in training if removing it
would leave an entity or relation without any training triple (the
transductive coverage constraint); kept partners are listed in the report.

Comparing a model on the original split and on its blocked variant then
measures the effect of direct recoverability on exactly the same test
triples.

Usage
    python make_blocked.py --src ../data/c2kg_N --out ../data/c2kg_N_blocked
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

from leakage_audit import (load, symmetry_rates, inverse_candidates,
                           coreferent_candidates)

COPY = ("entities.dict", "relations.dict", "valid.txt", "test.txt",
        "relation_patterns.json")


def partners(train, test, threshold=0.5, min_support=20):
    known = set(train)
    _, by_rel = symmetry_rates(train, min_support)
    inv_map, cor_map = defaultdict(set), defaultdict(set)
    for (a, b) in inverse_candidates(by_rel, threshold, min_support):
        inv_map[a].add(b)
    for (a, b) in coreferent_candidates(by_rel, threshold, min_support):
        cor_map[a].add(b)
        cor_map[b].add(a)
    out = {}
    for (h, r, t) in test:
        cand = [(t, r, h)]
        cand += [(t, r2, h) for r2 in inv_map.get(r, ())]
        cand += [(h, r2, t) for r2 in cor_map.get(r, ())]
        hit = [c for c in cand if c in known]
        if hit:
            out[(h, r, t)] = hit
    return out, dict(inv_map), {k: sorted(v) for k, v in cor_map.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-support", type=int, default=20)
    args = ap.parse_args()

    train = load(os.path.join(args.src, "train.txt"))
    valid = load(os.path.join(args.src, "valid.txt"))
    test = load(os.path.join(args.src, "test.txt"))
    leak, inv_map, cor_map = partners(train, test, args.threshold,
                                      args.min_support)
    remove = set(x for hit in leak.values() for x in hit)

    # coverage: count occurrences of every entity and relation in training
    ent_n, rel_n = Counter(), Counter()
    for h, r, t in train:
        ent_n[h] += 1
        ent_n[t] += 1
        rel_n[r] += 1
    kept = []
    for (h, r, t) in sorted(remove):
        if ent_n[h] <= 1 or ent_n[t] <= 1 or rel_n[r] <= 1:
            kept.append((h, r, t))
            continue
        ent_n[h] -= 1
        ent_n[t] -= 1
        rel_n[r] -= 1
    remove -= set(kept)

    os.makedirs(args.out, exist_ok=True)
    for f in COPY:
        shutil.copyfile(os.path.join(args.src, f), os.path.join(args.out, f))
    with open(os.path.join(args.out, "train.txt"), "w", newline="\n") as f:
        for (h, r, t) in train:                     # original order kept
            if (h, r, t) not in remove:
                f.write(f"{h}\t{r}\t{t}\n")
    with open(os.path.join(args.out, "filter_extra.txt"), "w",
              newline="\n") as f:
        for (h, r, t) in sorted(remove):
            f.write(f"{h}\t{r}\t{t}\n")
    blocked = sum(1 for hit in leak.values()
                  if all(x in remove for x in hit))
    rep = {"source": os.path.basename(os.path.normpath(args.src)),
           "test_triples": len(test),
           "test_triples_with_partner_in_train": len(leak),
           "test_triples_fully_blocked": blocked,
           "train_triples_removed": len(remove),
           "partners_kept_for_coverage": [list(x) for x in kept],
           "train_before": len(train), "train_after": len(train) - len(remove),
           "inverse_pairs": {k: sorted(v) for k, v in inv_map.items()},
           "coreferent_pairs": cor_map,
           "removed_by_relation": dict(Counter(r for _, r, _ in remove))}
    with open(os.path.join(args.out, "blocked_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("partners_kept_for_coverage",)}, indent=2))


if __name__ == "__main__":
    main()
