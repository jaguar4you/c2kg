"""
leakage_audit.py
----------------
Quantifies how much of a knowledge-graph test set is recoverable from the
training graph by a one-hop lookup rather than by inference.

Four leakage modes are measured, each as "what fraction of held-out test
triples has a training triple that restates it":

  1. INVERSE PAIR     (h,r1,t) held out while (t,r2,h) is in train, where
                      r2 behaves as the inverse of r1.  This is the mode
                      WN18 -> WN18RR and FB15k -> FB15k-237 were built to
                      remove.
  2. SYMMETRIC MIRROR (h,r,t) held out while (t,r,h) is in train, for a
                      relation that is its own inverse.  Not removed by the
                      WN18RR/FB15k-237 fixes.
  3. CO-REFERENT EDGE (h,r1,t) held out while (h,r2,t) is in train: the same
                      entity pair under a second relation name.
  4. COMPOSITION      (x,r3,z) held out while (x,r1,y) and (y,r2,z) are in
                      train, for a rule r1 o r2 -> r3.  Only checked when
                      rules are supplied with --rule.

For a supplied rule, the reverse direction is also measured: training
conclusions of the rule can reveal its held-out premises.

  5. PREMISE r2       (y,r2,z) held out while (x,r1,y) and (x,r3,z) are in
                      train for some x (a two-hop path y -r1^-1-> x -r3-> z).
                      When r1 is functional (one r1 value per x), the premise
                      follows exactly.  If r2 is symmetric, the mirrored
                      premise (z,r2,y) is counted as well.
  6. PREMISE r1       (x,r1,y) held out while y is the only entity whose
                      training r2-neighbours (both directions if r2 is
                      symmetric) include every training r3-tail of x.

Modes 4 to 6 are inference over two hops, not one-hop restatement; they are
reported so that the share of the test set open to such inference is known.

Reported per relation and in aggregate.  Nothing is modified; this is a
measurement tool.

Interpreting the result
-----------------------
A high mirror rate is not automatically a defect: symmetry is a real
relational pattern and one may legitimately want a model to learn it.  The
argument that it is leakage is that the held-out triple is then solved by a
one-hop lookup rather than by inference, which is exactly the argument used
to deprecate WN18.  Whichever position you take, the rate should be reported
so that readers can tell how much of a score is inference and how much is
retrieval.

Usage
    # C2KG (tab-separated train/valid/test.txt in one directory)
    python leakage_audit.py --data ../data/c2kg_N
    python leakage_audit.py --data ../data/c2kg_L

    # a public benchmark
    python leakage_audit.py --data ../data/wn18rr_O
    python leakage_audit.py --data FB15k-237 --top 15

    # with a composition rule (C2KG: locatedAt o adjacentTo -> nearLocation)
    python leakage_audit.py --data ../data/c2kg_L \
        --rule locatedAt,adjacentTo,nearLocation
"""

import argparse
import os
from collections import defaultdict


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) == 3:
                out.append(tuple(parts))
    return out


def symmetry_rates(train, min_support):
    by_rel = defaultdict(set)
    for h, r, t in train:
        by_rel[r].add((h, t))
    rates = {}
    for r, pairs in by_rel.items():
        if len(pairs) < min_support:
            rates[r] = None
            continue
        rates[r] = sum(1 for (h, t) in pairs if (t, h) in pairs) / len(pairs)
    return rates, by_rel


def inverse_candidates(by_rel, threshold, min_support):
    """(r1, r2) such that most of r1's pairs appear reversed under r2."""
    out = {}
    rels = [r for r in by_rel if len(by_rel[r]) >= min_support]
    for r1 in rels:
        rev = {(t, h) for (h, t) in by_rel[r1]}
        for r2 in rels:
            if r1 == r2:
                continue
            inter = len(rev & by_rel[r2])
            rate = inter / len(rev)
            if rate >= threshold:
                out[(r1, r2)] = rate
    return out


def coreferent_candidates(by_rel, threshold, min_support):
    out = {}
    rels = [r for r in by_rel if len(by_rel[r]) >= min_support]
    for i, r1 in enumerate(rels):
        for r2 in rels[i + 1:]:
            a, b = by_rel[r1], by_rel[r2]
            rate = len(a & b) / min(len(a), len(b))
            if rate >= threshold:
                out[(r1, r2)] = rate
    return out


def premise_report(train, test, r1, r2, r3, r2_symmetric):
    """Modes 5 and 6: held-out premises of r1 o r2 -> r3 revealed by
    training conclusions (see the module docstring)."""
    fwd = defaultdict(lambda: defaultdict(set))
    for (h, r, t) in train:
        fwd[r][h].add(t)
    heads1 = fwd[r1]
    functional = sum(1 for x in heads1 if len(heads1[x]) == 1)
    # mode 5: (y, r2, z) from x -r1-> y and x -r3-> z
    by_y = defaultdict(set)            # y -> {x : (x, r1, y) in train}
    for x, ys in heads1.items():
        for y in ys:
            by_y[y].add(x)

    def follows(y, z):
        return any(z in fwd[r3][x] for x in by_y.get(y, ()))
    tst = [(y, z) for (y, r, z) in test if r == r2]
    direct = sum(1 for (y, z) in tst if follows(y, z))
    either = sum(1 for (y, z) in tst
                 if follows(y, z) or (r2_symmetric and follows(z, y)))
    if tst:
        sym_txt = (f"; {either} ({100*either/len(tst):.1f}%) including "
                   f"the mirrored premise, {r2} being symmetric"
                   if r2_symmetric else "")
        print(f"  premise {r2}: {direct}/{len(tst)} "
              f"({100*direct/len(tst):.1f}%) of held-out {r2} triples follow "
              f"from {r1}^-1 o {r3} in training{sym_txt} "
              f"[{r1} has one value for {functional}/{len(heads1)} heads]")
    # mode 6: (x, r1, y) with y the only entity consistent with x's r3-tails
    nb = defaultdict(set)
    for h, ts in fwd[r2].items():
        for t in ts:
            nb[h].add(t)
            if r2_symmetric:
                nb[t].add(h)
    cands = set(nb)
    tst1 = [(x, y) for (x, r, y) in test if r == r1]
    with_tails = unique = 0
    for (x, y) in tst1:
        tails = fwd[r3].get(x, set())
        if not tails:
            continue
        with_tails += 1
        ok = [c for c in cands if tails <= nb[c]]
        unique += ok == [y]
    if tst1:
        print(f"  premise {r1}: {unique}/{len(tst1)} "
              f"({100*unique/len(tst1):.1f}%) of held-out {r1} triples have "
              f"a unique object consistent with the training {r3} and {r2} "
              f"edges of their subject ({with_tails} subjects have {r3} "
              f"edges in training)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory holding train.txt / valid.txt / test.txt")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="detection threshold for symmetric / inverse / "
                         "co-referent relations (default 0.5)")
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--top", type=int, default=25,
                    help="max relations to list in the per-relation table")
    ap.add_argument("--rule", action="append", default=[],
                    help="composition rule 'r1,r2,r3' meaning r1 o r2 -> r3; "
                         "repeatable")
    ap.add_argument("--include-valid", action="store_true",
                    help="count a leak if the restating triple is in train "
                         "OR valid (the filtered protocol removes both)")
    args = ap.parse_args()

    train = load(os.path.join(args.data, "train.txt"))
    valid = load(os.path.join(args.data, "valid.txt"))
    test = load(os.path.join(args.data, "test.txt"))
    known = set(train) | (set(valid) if args.include_valid else set())

    ents = {e for (h, r, t) in train + valid + test for e in (h, t)}
    rels = {r for (h, r, t) in train + valid + test}
    print(f"dataset : {args.data}")
    print(f"entities {len(ents)}  relations {len(rels)}  "
          f"train {len(train)}  valid {len(valid)}  test {len(test)}\n")

    rates, by_rel = symmetry_rates(train, args.min_support)
    sym = {r: v for r, v in rates.items()
           if v is not None and v >= args.threshold}
    invs = inverse_candidates(by_rel, args.threshold, args.min_support)
    cor = coreferent_candidates(by_rel, args.threshold, args.min_support)

    print("--- detected structures (from the training graph) ---")
    if sym:
        print("  self-inverse (symmetric) relations:")
        for r, v in sorted(sym.items(), key=lambda x: -x[1]):
            print(f"    {r:40s} symmetry {v:.3f}")
    else:
        print("  self-inverse (symmetric) relations: none")
    if invs:
        print("  inverse relation pairs:")
        seen = set()
        for (a, b), v in sorted(invs.items(), key=lambda x: -x[1]):
            if (b, a) in seen:
                continue
            seen.add((a, b))
            print(f"    {a} <-> {b:30s} {v:.3f}")
    else:
        print("  inverse relation pairs: none")
    if cor:
        print("  co-referent relation pairs:")
        for (a, b), v in sorted(cor.items(), key=lambda x: -x[1]):
            print(f"    {a} == {b:32s} overlap {v:.3f}")
    else:
        print("  co-referent relation pairs: none")

    inv_map = defaultdict(set)
    for (a, b) in invs:
        inv_map[a].add(b)
    cor_map = defaultdict(set)
    for (a, b) in cor:
        cor_map[a].add(b)
        cor_map[b].add(a)

    # ---------------- per-triple audit ----------------
    per_rel = defaultdict(lambda: {"n": 0, "mirror": 0, "inv": 0, "cor": 0})
    for (h, r, t) in test:
        d = per_rel[r]
        d["n"] += 1
        if (t, r, h) in known:
            d["mirror"] += 1
        if any((t, r2, h) in known for r2 in inv_map.get(r, ())):
            d["inv"] += 1
        if any((h, r2, t) in known for r2 in cor_map.get(r, ())):
            d["cor"] += 1

    print("\n--- per-relation leakage in the test split ---")
    print(f"{'relation':40s} {'test':>6s} {'mirror':>14s} {'inverse':>12s}"
          f" {'co-ref':>12s}")
    rows = sorted(per_rel.items(), key=lambda x: -x[1]["n"])[:args.top]
    for r, d in rows:
        n = d["n"]
        def pct(k):
            return f"{d[k]:5d} ({100*d[k]/n:5.1f}%)" if n else "    -"
        flag = " <<<" if n and max(d["mirror"], d["inv"], d["cor"]) / n > 0.2 else ""
        print(f"{r:40s} {n:6d} {pct('mirror'):>14s} {pct('inv'):>12s}"
              f" {pct('cor'):>12s}{flag}")

    tot = len(test)
    any_leak = sum(1 for (h, r, t) in test
                   if (t, r, h) in known
                   or any((t, r2, h) in known for r2 in inv_map.get(r, ()))
                   or any((h, r2, t) in known for r2 in cor_map.get(r, ())))
    m = sum(d["mirror"] for d in per_rel.values())
    i = sum(d["inv"] for d in per_rel.values())
    c = sum(d["cor"] for d in per_rel.values())
    print(f"\n--- aggregate over {tot} test triples ---")
    print(f"  symmetric mirror : {m:6d} ({100*m/tot:5.1f}%)")
    print(f"  inverse pair     : {i:6d} ({100*i/tot:5.1f}%)")
    print(f"  co-referent edge : {c:6d} ({100*c/tot:5.1f}%)")
    print(f"  ANY of the above : {any_leak:6d} ({100*any_leak/tot:5.1f}%)")

    # ---------------- composition ----------------
    if args.rule:
        print("\n--- composition leakage (supplied rules) ---")
        out_by = defaultdict(list)
        for (h, r, t) in train:
            out_by[(h, r)].append(t)
        for spec in args.rule:
            r1, r2, r3 = [s.strip() for s in spec.split(",")]
            tst = [(h, r, t) for (h, r, t) in test if r == r3]
            der = 0
            for (x, _, z) in tst:
                ys = out_by.get((x, r1), ())
                if any(z in out_by.get((y, r2), ()) for y in ys):
                    der += 1
            if tst:
                print(f"  {r1} o {r2} -> {r3}: {der}/{len(tst)} "
                      f"({100*der/len(tst):.1f}%) of held-out {r3} triples "
                      f"are derivable from training edges")
            else:
                print(f"  {r1} o {r2} -> {r3}: no {r3} triples in test")
            premise_report(train, test, r1, r2, r3, r2 in sym)

    print("\nNote: a high mirror rate is a property of the split, not proof "
          "of a defect.\nReport it and let the reader judge how much of the "
          "score is retrieval.")


if __name__ == "__main__":
    main()
