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

  rule    A path-rule miner in the style of AnyBURL (without rules that
          contain constants).  It mines every rule whose body is one or two
          edges, each traversable in either direction,
              r1 -> r3   and   r1 o r2 -> r3,   r1, r2 in {r, r^-1},
          from the training graph; the confidence of a rule is the number
          of its body groundings (x, z) that are training triples (x, r3, z)
          divided by the number of body groundings, and rules with fewer than
          --min-support such triples are dropped.  A candidate z of a query
          (x, r3, ?) collects the confidences of the rules that reach it
          from x (the query entity itself is never proposed), aggregated in
          one of three ways:
              max            the highest confidence
              noisy-or       1 - prod(1 - conf) over the rules that reach z
              noisy-or-paths 1 - prod(1 - conf) over every path that reaches z
          The aggregation and a minimum confidence (0, 0.01 or 0.1) are
          selected on the validation tail MRR, as the learning rates of the
          trained models are; the test set enters no decision.  Unlike
          ``path``, the rules are discovered, not supplied.

  type    Type-constrained random ranking (C2KG only): every entity of the
          range type of r (ontology.py) scores 1, every other entity 0, so
          the gold entity is ranked uniformly at random among the filtered
          candidates of the right type.  This is the level a method reaches
          on a relation whose held-out triples carry no signal beyond the
          entity types.

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
    python path_probe.py --data ../data/wn18rr_L --probe rule
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
    """Path rules of length 1 and 2 with inverse edges (see the module
    docstring).  Iteration orders are fixed (sorted), so the scores, and
    hence the ties, are reproducible bit for bit."""
    name = "rule"
    AGGS = ("max", "noisy-or", "noisy-or-paths")
    MIN_CONFS = (0.0, 0.01, 0.1)

    def __init__(self, train, min_support=2):
        out = defaultdict(set)
        for h, r, t in train:
            out[(h, r)].add(t)
            out[(t, r + "^-1")].add(h)
        self.out = {k: sorted(v) for k, v in out.items()}
        rels = sorted({r for _, r, _ in train})
        by_rel = defaultdict(set)
        for h, r, t in train:
            by_rel[r].add((h, t))
        rels_at = defaultdict(list)                  # node -> its edge labels
        for (x, r) in sorted(self.out):
            rels_at[x].append(r)
        mined = []
        # length 1
        pairs1 = defaultdict(set)
        for (x, r1), zs in self.out.items():
            for z in zs:
                pairs1[r1].add((x, z))
        for r1 in sorted(pairs1):
            for r3 in rels:
                if r1 == r3:
                    continue
                hits = len(pairs1[r1] & by_rel[r3])
                if hits >= min_support:
                    mined.append((hits / len(pairs1[r1]), hits, (r1,), r3))
        # length 2
        body = defaultdict(set)
        for (x, r1), ys in self.out.items():
            for y in ys:
                for r2 in rels_at[y]:
                    for z in self.out[(y, r2)]:
                        if z != x:
                            body[(r1, r2)].add((x, z))
        for b in sorted(body):
            for r3 in rels:
                hits = len(body[b] & by_rel[r3])
                if hits >= min_support:
                    mined.append((hits / len(body[b]), hits, b, r3))
        mined.sort(key=lambda m: (-m[0], -m[1], m[2], m[3]))
        self.mined = mined
        self.by_head = defaultdict(list)
        for conf, hits, b, r3 in mined:
            self.by_head[r3].append((conf, b))
        self.agg, self.min_conf = "noisy-or-paths", 0.0
        self.selection = []

    def score_tails(self, h, r):
        sc = {}
        paths = self.agg == "noisy-or-paths"
        for conf, b in self.by_head.get(r, ()):
            if conf < self.min_conf:
                continue
            if len(b) == 1:
                zs = self.out.get((h, b[0]), ())
            else:
                zs = [z for y in self.out.get((h, b[0]), ())
                      for z in self.out.get((y, b[1]), ())]
                if not paths:
                    zs = sorted(set(zs))
            for z in zs:
                if z == h:
                    continue
                if self.agg == "max":
                    sc[z] = max(sc.get(z, 0.0), conf)
                else:
                    sc[z] = 1.0 - (1.0 - sc.get(z, 0.0)) * (1.0 - conf)
        return sc

    def select(self, valid, all_true, n_ent, patterns):
        """Choose the aggregation and the minimum confidence on the
        validation tail MRR (first best in the order of the grid)."""
        best = None
        for agg in self.AGGS:
            for c in self.MIN_CONFS:
                self.agg, self.min_conf = agg, c
                v = evaluate(self, valid, all_true, n_ent, patterns)[0]["MRR"]
                self.selection.append({"aggregation": agg, "min_conf": c,
                                       "valid_tail_MRR": v})
                if best is None or v > best[0]:
                    best = (v, agg, c)
        self.agg, self.min_conf = best[1], best[2]
        return {"aggregation": best[1], "min_conf": best[2],
                "valid_tail_MRR": best[0]}

    def report(self, top=15, min_hits=10):
        lines = [f"mined {len(self.mined)} rules; highest confidence "
                 f"(at least {min_hits} supporting triples):"]
        shown = [m for m in self.mined if m[1] >= min_hits][:top]
        for conf, hits, b, r3 in shown:
            lines.append(f"    {conf:5.3f}  n={hits:5d}   "
                         f"{' o '.join(b)} -> {r3}")
        if not shown:
            lines.append("    (none)")
        return "\n".join(lines)


class TypeProbe:
    """Uniform ranking among the entities of the range type of r."""
    name = "type"

    def __init__(self, entities):
        from ontology import RELATION_DOMAIN_RANGE, entity_type
        by_type = defaultdict(list)
        for e in sorted(entities):
            by_type[entity_type(e)].append(e)
        self.cands = {r: {e: 1.0 for t in sorted(rng) for e in by_type[t]}
                      for r, (_, rng) in RELATION_DOMAIN_RANGE.items()}

    @staticmethod
    def applies(relations):
        from ontology import RELATION_DOMAIN_RANGE
        return set(relations) <= set(RELATION_DOMAIN_RANGE)

    def score_tails(self, h, r):
        return self.cands[r]


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
    recs, gm = [], []
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
        gm.append((g, m))
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

    per_triple = {"tail_g": [x[0] for x in gm], "tail_m": [x[1] for x in gm]}
    return agg(recs), {p: agg(v) for p, v in per_pat.items()}, per_triple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probe", default="all",
                    choices=["path", "rule", "type", "freq", "all"])
    ap.add_argument("--rule", action="append", default=[],
                    help="composition rule 'r1,r2,r3'; repeatable")
    ap.add_argument("--min-support", type=int, default=2,
                    help="rule probe: minimum number of supporting training "
                         "triples of a rule (default 2)")
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
        rp = RuleProbe(train, args.min_support)
        print(rp.report())
        sel = rp.select(valid, all_true, n_ent, patterns)
        for x in rp.selection:
            print(f"    validation: {x['aggregation']:15s} min_conf "
                  f"{x['min_conf']:<5}  tail MRR {x['valid_tail_MRR']:.4f}")
        print(f"  selected on validation: {sel['aggregation']}, min_conf "
              f"{sel['min_conf']}\n")
        probes.append(rp)
    if args.probe in ("type", "all"):
        if TypeProbe.applies({r for _, r, _ in train + valid + test}):
            probes.append(TypeProbe(ents))
        else:
            print("[type] skipped: relations not in ontology.py (C2KG only)")
    if args.probe in ("freq", "all"):
        probes.append(FreqProbe(train))

    results = {}
    for pr in probes:
        overall, per_pat, per_triple = evaluate(pr, test, all_true, n_ent,
                                                patterns)
        results[pr.name] = {"overall": overall, "per_pattern": per_pat,
                            "per_triple": per_triple}
        if pr.name == "rule":
            results["rule"]["selected"] = sel
            results["rule"]["validation"] = rp.selection
            results["rule"]["n_rules"] = len(rp.mined)
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
