"""
split_fixed.py
--------------
Leakage-aware splitting for knowledge-graph completion benchmarks.

The grouped split in ``data_gen.py`` co-assigns explicit inverse pairs
(threatens/threatenedBy, supports/supportedBy) so that a held-out triple
cannot be solved by memorising its counterpart in training.  That rule is
correct but incomplete: it covers only one of the ways a held-out triple can
be a restatement of a training triple.  This module closes the two that
remain.

  (a) INVERSE PAIR      (h, r1, t) and (t, r2, h) where r2 is inverse of r1
                        -- already handled by data_gen.py
  (b) SYMMETRIC MIRROR  (h, r, t) and (t, r, h) where r is its own inverse
                        -- e.g. adjacentTo, alliedWith in C2KG, or
                        _derivationally_related_form in WN18RR.
  (c) CO-REFERENT EDGE  (h, r1, t) and (h, r2, t): the same entity pair
                        asserted under two relation names.  In C2KG the
                        generator writes occursAt and locatedAt to the same
                        (engagement, location) pair for every engagement, so
                        one is a free copy of the other.

All three are handled by the same mechanism: build the equivalence classes
of mutually derivable triples with a union-find, then assign whole classes
to a split atomically.

Composition leakage -- (x, r3, z) derivable from (x, r1, y) and (y, r2, z) --
is NOT fixed here.  Grouping cannot fix it without cascading: the closure of
a 2-hop rule pulls in chains that quickly swallow the whole graph.  It is
reported as a diagnostic instead (see ``leakage_audit.py``) so that the
residual can be stated honestly rather than silently absorbed.

Usage
    python split_fixed.py --out ../data/c2kg_L --seed 42

    from split_fixed import split_triples_leakage_aware
    train, valid, test, report = split_triples_leakage_aware(
        triples, seed=42,
        inverse_pairs=(("threatens","threatenedBy"),
                       ("supports","supportedBy")))
    print(report)
"""

from collections import defaultdict


# --------------------------------------------------------------------------
# union-find
# --------------------------------------------------------------------------
class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# --------------------------------------------------------------------------
# automatic detection of the leaky relation structures
# --------------------------------------------------------------------------
def detect_symmetric(triples, threshold=0.5, min_support=20):
    """Relations for which (h,r,t) and (t,r,h) co-occur at least
    ``threshold`` of the time.  Such a relation is its own inverse, so a
    held-out triple is recoverable from its mirror."""
    by_rel = defaultdict(set)
    for h, r, t in triples:
        by_rel[r].add((h, t))
    out = {}
    for r, pairs in by_rel.items():
        if len(pairs) < min_support:
            continue
        rate = sum(1 for (h, t) in pairs if (t, h) in pairs) / len(pairs)
        if rate >= threshold:
            out[r] = rate
    return out


def detect_coreferent(triples, threshold=0.5, min_support=20):
    """Ordered relation pairs (r1, r2) such that a large fraction of r1's
    entity pairs also carry r2.  One is then a free copy of the other."""
    by_rel = defaultdict(set)
    for h, r, t in triples:
        by_rel[r].add((h, t))
    out = {}
    rels = list(by_rel)
    for i, r1 in enumerate(rels):
        for r2 in rels[i + 1:]:
            a, b = by_rel[r1], by_rel[r2]
            if len(a) < min_support or len(b) < min_support:
                continue
            inter = len(a & b)
            rate = inter / min(len(a), len(b))
            if rate >= threshold:
                out[(r1, r2)] = rate
    return out


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------
def build_groups(triples, inverse_pairs=(), symmetric=None, coreferent=None,
                 auto=True, sym_threshold=0.5, coref_threshold=0.5):
    """Equivalence classes of mutually derivable triples."""
    triples = list(triples)
    tset = set(triples)

    if auto:
        symmetric = dict(symmetric or {})
        symmetric.update(detect_symmetric(triples, sym_threshold))
        coreferent = dict(coreferent or {})
        coreferent.update(detect_coreferent(triples, coref_threshold))
    else:
        symmetric = dict(symmetric or {})
        coreferent = dict(coreferent or {})

    inv = {}
    for a, b in inverse_pairs:
        inv[a] = b
        inv[b] = a

    dsu = _DSU()
    for tr in triples:
        dsu.find(tr)

    for (h, r, t) in triples:
        # (a) explicit inverse pair
        if r in inv:
            mate = (t, inv[r], h)
            if mate in tset:
                dsu.union((h, r, t), mate)
        # (b) symmetric mirror (self-inverse relation)
        if r in symmetric:
            mate = (t, r, h)
            if mate in tset:
                dsu.union((h, r, t), mate)

    # (c) co-referent edges: same (h,t) under two relation names
    for (r1, r2) in coreferent:
        for (h, r, t) in triples:
            if r == r1 and (h, r2, t) in tset:
                dsu.union((h, r1, t), (h, r2, t))

    groups = defaultdict(list)
    for tr in triples:
        groups[dsu.find(tr)].append(tr)
    return list(groups.values()), symmetric, coreferent


def split_triples_leakage_aware(triples, seed=42, valid_ratio=0.10,
                                test_ratio=0.10, inverse_pairs=(),
                                auto=True, verbose=True):
    """Leakage-aware grouped split with the same coverage guarantee as
    ``data_gen.split_triples_grouped`` (every entity and relation keeps at
    least one training occurrence)."""
    import random
    rng = random.Random(seed)
    triples = list(triples)

    groups, symmetric, coreferent = build_groups(
        triples, inverse_pairs=inverse_pairs, auto=auto)
    rng.shuffle(groups)

    ent_count, rel_count = defaultdict(int), defaultdict(int)
    for h, r, t in triples:
        ent_count[h] += 1
        ent_count[t] += 1
        rel_count[r] += 1

    n = len(triples)
    n_test, n_valid = int(test_ratio * n), int(valid_ratio * n)
    train, valid, test = [], [], []
    ec, rc = defaultdict(int), defaultdict(int)

    def safe(grp):
        eh, rh = defaultdict(int), defaultdict(int)
        for h, r, t in grp:
            eh[h] += 1
            eh[t] += 1
            rh[r] += 1
        return (all(ent_count[e] - ec[e] - eh[e] > 0 for e in eh) and
                all(rel_count[r] - rc[r] - rh[r] > 0 for r in rh))

    def push(dst, grp):
        dst.extend(grp)
        for h, r, t in grp:
            ec[h] += 1
            ec[t] += 1
            rc[r] += 1

    for grp in groups:
        if len(test) + len(grp) <= n_test and safe(grp):
            push(test, grp)
        elif len(valid) + len(grp) <= n_valid and safe(grp):
            push(valid, grp)
        else:
            push(train, grp)

    sizes = defaultdict(int)
    for g in groups:
        sizes[len(g)] += 1
    report = {
        "symmetric_relations": symmetric,
        "coreferent_relation_pairs": coreferent,
        "n_groups": len(groups),
        "group_size_histogram": dict(sorted(sizes.items())),
        "split": (len(train), len(valid), len(test)),
    }
    if verbose:
        print("[split_fixed] symmetric relations detected:")
        for r, v in sorted(symmetric.items(), key=lambda x: -x[1]):
            print(f"    {r:32s} symmetry rate {v:.3f}")
        print("[split_fixed] co-referent relation pairs detected:")
        for (a, b), v in sorted(coreferent.items(), key=lambda x: -x[1]):
            print(f"    {a} == {b:24s} overlap {v:.3f}")
        print(f"[split_fixed] {len(groups)} groups, sizes "
              f"{report['group_size_histogram']}")
        print(f"[split_fixed] train/valid/test = {report['split']}")
    return train, valid, test, report


# --------------------------------------------------------------------------
# CLI: generate C2KG with the leakage-aware split (C2KG-L)
# --------------------------------------------------------------------------
def main():
    import os, json, argparse
    from data_gen import build_graph, REL_PATTERNS, INVERSE_PAIRS

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/c2kg_L")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-auto", action="store_true",
                    help="disable automatic detection; group explicit "
                         "inverse pairs only (reproduces the old behaviour)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    triples, relations = build_graph(seed=args.seed)
    train, valid, test, rep = split_triples_leakage_aware(
        triples, seed=args.seed, inverse_pairs=INVERSE_PAIRS,
        auto=not args.no_auto)

    entities = sorted({e for (h, r, t) in triples for e in (h, t)})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    def dump(name, rows):
        with open(os.path.join(args.out, name), "w") as f:
            for h, r, t in rows:
                f.write(f"{h}\t{r}\t{t}\n")

    dump("train.txt", train)
    dump("valid.txt", valid)
    dump("test.txt", test)
    with open(os.path.join(args.out, "entities.dict"), "w") as f:
        for e, i in ent2id.items():
            f.write(f"{i}\t{e}\n")
    with open(os.path.join(args.out, "relations.dict"), "w") as f:
        for r, i in rel2id.items():
            f.write(f"{i}\t{r}\n")
    with open(os.path.join(args.out, "relation_patterns.json"), "w") as f:
        json.dump(REL_PATTERNS, f, indent=2)
    with open(os.path.join(args.out, "split_report.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, dict) else
                       {str(kk): vv for kk, vv in v.items()})
                   for k, v in rep.items()}, f, indent=2)

    print(f"\nEntities : {len(entities)}")
    print(f"Relations: {len(relations)}")
    print(f"Triples  : total={len(triples)} train={len(train)} "
          f"valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
