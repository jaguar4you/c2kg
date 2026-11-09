"""
path_probe.py
-------------
Training-free probes for the composition-pattern gap.

Purpose
-------
C2KG contains a composition rule (``locatedAt`` o ``adjacentTo`` ->
``nearLocation``), and part of the held-out ``nearLocation`` triples can be
derived in two hops from edges that remain in training.  These probes
measure how far simple, parameter-free procedures get on the benchmark, so
that trained models can be compared against a lookup floor and a rule
baseline.  They have no parameters and no training.

Probes
------
  path    For a query (x, r3, ?), score each candidate z by the number of
          2-hop paths x -r1-> y -r2-> z, for rules supplied with --rule.
          Pure lookup: no learning, no embeddings.

  rule    A minimal AnyBURL-style miner.  Mines all 2-hop rules
          r1 o r2 -> r3 from the training graph, keeps those whose
          confidence exceeds --min-conf with at least --min-support
          groundings, then scores candidates by the best rule confidence.
          Unlike ``path``, the rules are discovered, not supplied, so this
          also reports whether the intended composition rule is the one a
          miner would actually find.

  freq    Frequency baseline: score every candidate by how often it appears
          as a tail of that relation in training.  This is the "by-copy"
          rule of Ferre (2021) and the sanity floor -- any model that does
          not beat it on a relation has learned nothing about that relation.

Evaluation uses the same filtered protocol as train_eval.py (all other true
triples removed from the candidate list before ranking, exact expectations
under random tie-breaking) and scores tail queries only, so the numbers are directly
comparable with the tail-only MRR of the trained models.

Usage
    python path_probe.py --data ../data/c2kg_L \
        --rule locatedAt,adjacentTo,nearLocation
    python path_probe.py --data ../data/c2kg_L --probe rule
    python path_probe.py --data ../data/c2kg_L --probe freq
    python path_probe.py --data ../data/c2kg_L --probe all \
        --rule locatedAt,adjacentTo,nearLocation --out ../results/probe_L.json
"""

import argparse
import json
import os
from collections import defaultdict


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_triples(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split("\t")
            if len(p) != 3:
                p = line.split()
            if len(p) == 3:
                out.append(tuple(p))
    return out


def load_patterns(data_dir):
    p = os.path.join(data_dir, "relation_patterns.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# --------------------------------------------------------------------------
# probes -- each returns a dict {candidate_entity: score} for a tail query
# --------------------------------------------------------------------------
class PathProbe:
    """Score by the number of 2-hop groundings of supplied rules."""
    name = "path"

    def __init__(self, train, rules):
        self.rules_by_head = defaultdict(list)   # r3 -> [(r1, r2), ...]
        for r1, r2, r3 in rules:
            self.rules_by_head[r3].append((r1, r2))
        self.out = defaultdict(list)             # (h, r) -> [t, ...]
        for h, r, t in train:
            self.out[(h, r)].append(t)

    def score_tails(self, h, r):
        sc = defaultdict(float)
        for (r1, r2) in self.rules_by_head.get(r, ()):
            for y in self.out.get((h, r1), ()):
                for z in self.out.get((y, r2), ()):
                    sc[z] += 1.0
        return sc


class RuleProbe:
    """Mine 2-hop rules r1 o r2 -> r3 and score by best rule confidence."""
    name = "rule"

    def __init__(self, train, min_conf=0.1, min_support=5, max_rules=200):
        self.out = defaultdict(list)
        for h, r, t in train:
            self.out[(h, r)].append(t)
        rels = sorted({r for _, r, _ in train})
        tset = set(train)

        # body groundings: for each (r1, r2), collect (x, z) reachable in 2 hops
        body = defaultdict(set)
        for (x, r1), ys in list(self.out.items()):
            for y in ys:
                for r2 in rels:
                    for z in self.out.get((y, r2), ()):
                        if z != x:
                            body[(r1, r2)].add((x, z))

        mined = []
        for (r1, r2), pairs in body.items():
            if len(pairs) < min_support:
                continue
            for r3 in rels:
                hits = sum(1 for (x, z) in pairs if (x, r3, z) in tset)
                if hits < min_support:
                    continue
                conf = hits / len(pairs)
                if conf >= min_conf:
                    mined.append((conf, hits, r1, r2, r3))
        mined.sort(reverse=True)
        self.mined = mined[:max_rules]
        self.by_head = defaultdict(list)
        for conf, hits, r1, r2, r3 in self.mined:
            self.by_head[r3].append((conf, r1, r2))

    def score_tails(self, h, r):
        sc = defaultdict(float)
        for (conf, r1, r2) in self.by_head.get(r, ()):
            for y in self.out.get((h, r1), ()):
                for z in self.out.get((y, r2), ()):
                    if conf > sc[z]:
                        sc[z] = conf
        return sc

    def report(self, top=15):
        lines = ["mined 2-hop rules (confidence, support, rule):"]
        for conf, hits, r1, r2, r3 in self.mined[:top]:
            lines.append(f"    {conf:5.3f}  n={hits:5d}   "
                         f"{r1} o {r2} -> {r3}")
        if not self.mined:
            lines.append("    (none above threshold)")
        return "\n".join(lines)


class FreqProbe:
    """Score by training frequency of the candidate as a tail of r."""
    name = "freq"

    def __init__(self, train):
        self.cnt = defaultdict(lambda: defaultdict(float))
        for h, r, t in train:
            self.cnt[r][t] += 1.0

    def score_tails(self, h, r):
        return self.cnt.get(r, {})


# --------------------------------------------------------------------------
# filtered evaluation (tail queries; head queries need inverse edges, and a
# lookup probe is direction-specific, so we report tail-side only and say so)
# --------------------------------------------------------------------------
def _expected(g, m, k):
    """Exact expectations for a gold entity with g candidates above it and
    m candidates (itself included) tied with it, under random tie-breaking
    (the same definitions as train_eval.py)."""
    rr = sum(1.0 / (g + j) for j in range(1, m + 1)) / m
    hk = max(0, min(m, k - g)) / m
    return rr, hk


def evaluate(probe, test, all_true_tails, n_ent_names, patterns):
    """Filtered MRR / Hits@K over tail queries, overall and per pattern.

    Every candidate the probe does not score is treated as score 0, so the
    gold entity of a query the probe cannot answer ties with all other
    unscored candidates and receives the random-ranking value."""
    per_pat = defaultdict(list)
    recs = []
    for (h, r, t) in test:
        sc = probe.score_tails(h, r)
        gold = sc.get(t, 0.0)
        # filtered: ignore other true tails for this (h, r); `others`
        # contains the gold itself
        others = all_true_tails.get((h, r), set())
        if gold <= 0.0:
            # unscored gold: ties with every unscored, unfiltered candidate
            g = sum(1 for e, s in sc.items()
                    if s > 0 and e != t and e not in others)
            m = n_ent_names - g - len(others - {t})       # gold included
        else:
            g = sum(1 for e, s in sc.items()
                    if s > gold and e != t and e not in others)
            m = 1 + sum(1 for e, s in sc.items()
                        if s == gold and e != t and e not in others)
        rr, h1 = _expected(g, m, 1)
        _, h10 = _expected(g, m, 10)
        rec = (rr, h1, h10, g + (m + 1) / 2.0)
        recs.append(rec)
        if patterns:
            per_pat[patterns.get(r, "unknown")].append(rec)

    def agg(rs):
        n = len(rs)
        if not n:
            return None
        return {"MRR": sum(x[0] for x in rs) / n,
                "H@1": sum(x[1] for x in rs) / n,
                "H@10": sum(x[2] for x in rs) / n,
                "MR": sum(x[3] for x in rs) / n, "n": n}

    return agg(recs), {p: agg(v) for p, v in per_pat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probe", default="all",
                    choices=["path", "rule", "freq", "all"])
    ap.add_argument("--rule", action="append", default=[],
                    help="composition rule 'r1,r2,r3'; repeatable")
    ap.add_argument("--min-conf", type=float, default=0.1)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = load_triples(os.path.join(args.data, "train.txt"))
    valid = load_triples(os.path.join(args.data, "valid.txt"))
    test = load_triples(os.path.join(args.data, "test.txt"))
    xp = os.path.join(args.data, "filter_extra.txt")    # leak-blocked data
    extra = load_triples(xp) if os.path.exists(xp) else []
    patterns = load_patterns(args.data)
    ents = {e for (h, r, t) in train + valid + test + extra for e in (h, t)}
    n_ent = len(ents)

    all_true = defaultdict(set)
    for h, r, t in train + valid + test + extra:
        all_true[(h, r)].add(t)

    print(f"dataset {args.data}: {n_ent} entities, "
          f"train {len(train)}, test {len(test)}")
    print("NOTE: tail-side queries only; compare with the tail-only MRR "
          "of the trained models.\n")

    rules = []
    for spec in args.rule:
        r1, r2, r3 = [s.strip() for s in spec.split(",")]
        rules.append((r1, r2, r3))

    probes = []
    if args.probe in ("path", "all"):
        if not rules:
            print("[path] skipped: no --rule supplied")
        else:
            probes.append(PathProbe(train, rules))
    if args.probe in ("rule", "all"):
        rp = RuleProbe(train, args.min_conf, args.min_support)
        print(rp.report())
        print()
        probes.append(rp)
    if args.probe in ("freq", "all"):
        probes.append(FreqProbe(train))

    results = {}
    for pr in probes:
        overall, per_pat = evaluate(pr, test, all_true, n_ent, patterns)
        results[pr.name] = {"overall": overall, "per_pattern": per_pat}
        print(f"=== probe: {pr.name} (no training) ===")
        print(f"  overall   MRR {overall['MRR']:.4f}  "
              f"H@1 {overall['H@1']:.4f}  H@10 {overall['H@10']:.4f}  "
              f"(n={overall['n']})")
        for p in ("symmetric", "antisymmetric", "inverse", "composition",
                  "N-N", "general"):
            a = per_pat.get(p)
            if a:
                print(f"  {p:14s} MRR {a['MRR']:.4f}  H@1 {a['H@1']:.4f}  "
                      f"H@10 {a['H@10']:.4f}  (n={a['n']})")
        print()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
