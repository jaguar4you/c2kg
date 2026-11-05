"""
wn18rr_prepare.py
-----------------
Builds the two WN18RR datasets of the WN18RR experiment in the C2KG data
format.

  wn18rr_O   the original split of Dettmers et al. (2018), unchanged
  wn18rr_L   the same 93,003 triples re-split leakage-aware, with exactly the
             original sizes (86,835 / 3,034 / 3,134)

The re-split uses the procedure of C2KG-L (split_fixed.build_groups): a
union-find merges every triple with its symmetric mirror and its co-referent
copies (structures detected from the data, threshold 0.5, support 20), and
whole groups are assigned to test, then validation, under the constraint that
every entity and relation keeps at least one training triple.  WN18RR has no
inverse relation pairs (they were removed when it was built from WN18).

Each output folder holds entities.dict, relations.dict, train/valid/test.txt
and relation_patterns.json (detected symmetric relations -> "symmetric",
all others -> "general"), so that c2kg_runner.py / wn18rr_runner.py can read
it unchanged.  Both folders share one entities.dict and relations.dict, so
entity and relation ids are identical across the two splits.

The input files are checked against the standard WN18RR by content (sorted
triples), so line endings and line order do not matter.

Usage (from the code folder)
    python wn18rr_prepare.py --src ../data/wn18rr_original --out-root ../data
"""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict

from split_fixed import build_groups

# content hashes of the standard WN18RR (sorted "h\tr\tt" lines, '\n'-joined);
# identical for the copies in the RotatE repository
# (DeepGraphLearning/KnowledgeGraphEmbedding, data/wn18rr) and in
# villmow/datasets_knowledge_embedding (WN18RR/original)
REFERENCE = {
    "train": (86835, "328be176010cc13057da5d3bcafabb256d3b1d22ce9ce96e268c65f29dda6c5b"),
    "valid": (3034, "90a1ed6adf244953bfef0d24766b6a44a32e11d9b9c328687a0dcfced874cb41"),
    "test": (3134, "db0457bbb9ad4131c4f5cfb3143b43b6ff5c84d602a82529b2010d83b0d5e20c"),
}
N_ENT, N_REL = 40943, 11


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            if len(p) != 3:
                p = line.split()
            if len(p) != 3:
                raise SystemExit(f"{path}: cannot parse line {line!r}")
            out.append(tuple(p))
    return out


def content_hash(triples):
    s = "\n".join("\t".join(t) for t in sorted(triples))
    return hashlib.sha256(s.encode()).hexdigest()


def split_exact(triples, n_valid, n_test, seed):
    """Leakage-aware grouped split with exact validation and test sizes."""
    triples = sorted(triples)                    # independent of file order
    groups, symmetric, coreferent = build_groups(triples, auto=True)
    rng = random.Random(seed)
    rng.shuffle(groups)

    ent_count, rel_count = defaultdict(int), defaultdict(int)
    for h, r, t in triples:
        ent_count[h] += 1
        ent_count[t] += 1
        rel_count[r] += 1
    ec, rc = defaultdict(int), defaultdict(int)
    train, valid, test = [], [], []

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
    return train, valid, test, {
        "symmetric_relations": symmetric,
        "coreferent_relation_pairs": {f"{a}|{b}": v
                                      for (a, b), v in coreferent.items()},
        "n_groups": len(groups),
        "group_size_histogram": dict(sorted(sizes.items())),
    }


def write_dataset(folder, train, valid, test, ent2id, rel2id, patterns):
    os.makedirs(folder, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        with open(os.path.join(folder, f"{name}.txt"), "w",
                  encoding="utf-8", newline="\n") as f:
            for h, r, t in rows:
                f.write(f"{h}\t{r}\t{t}\n")
    with open(os.path.join(folder, "entities.dict"), "w",
              encoding="utf-8", newline="\n") as f:
        for e, i in ent2id.items():
            f.write(f"{i}\t{e}\n")
    with open(os.path.join(folder, "relations.dict"), "w",
              encoding="utf-8", newline="\n") as f:
        for r, i in rel2id.items():
            f.write(f"{i}\t{r}\n")
    with open(os.path.join(folder, "relation_patterns.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(patterns, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder with the original WN18RR train/valid/test.txt")
    ap.add_argument("--out-root", default="../data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    orig = {k: load(os.path.join(args.src, f"{k}.txt"))
            for k in ("train", "valid", "test")}

    # ---- check that this is the standard WN18RR ---------------------------
    problems = []
    for k, rows in orig.items():
        n_ref, h_ref = REFERENCE[k]
        if len(rows) != n_ref:
            problems.append(f"{k}: {len(rows)} triples, expected {n_ref}")
        if len(set(rows)) != len(rows):
            problems.append(f"{k}: contains duplicate triples")
        if h_ref is not None and content_hash(rows) != h_ref:
            problems.append(f"{k}: content differs from the standard WN18RR")
    allt = orig["train"] + orig["valid"] + orig["test"]
    if len(set(allt)) != len(allt):
        problems.append("a triple occurs in more than one split")
    ents = sorted({e for h, _, t in allt for e in (h, t)})
    rels = sorted({r for _, r, _ in allt})
    if len(ents) != N_ENT or len(rels) != N_REL:
        problems.append(f"{len(ents)} entities / {len(rels)} relations, "
                        f"expected {N_ENT} / {N_REL}")
    if problems:
        raise SystemExit("STOPPED: input is not the standard WN18RR:\n  "
                         + "\n  ".join(problems))
    print(f"input ok: standard WN18RR, {N_ENT} entities, {N_REL} relations, "
          f"train/valid/test = {len(orig['train'])}/{len(orig['valid'])}/"
          f"{len(orig['test'])}")

    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}

    # ---- leakage-aware re-split ------------------------------------------
    train, valid, test, rep = split_exact(
        allt, len(orig["valid"]), len(orig["test"]), args.seed)
    sizes = (len(train), len(valid), len(test))
    want = (len(orig["train"]), len(orig["valid"]), len(orig["test"]))
    if sizes != want:
        raise SystemExit(f"STOPPED: re-split sizes {sizes}, expected {want}")
    if set(train) | set(valid) | set(test) != set(allt):
        raise SystemExit("STOPPED: re-split lost or added triples")
    tr_ents = {e for h, _, t in train for e in (h, t)}
    tr_rels = {r for _, r, _ in train}
    if len(tr_ents) != N_ENT or len(tr_rels) != N_REL:
        raise SystemExit("STOPPED: re-split training set misses an entity "
                         "or relation")

    patterns = {r: ("symmetric" if r in rep["symmetric_relations"]
                    else "general") for r in rels}

    out_o = os.path.join(args.out_root, "wn18rr_O")
    out_l = os.path.join(args.out_root, "wn18rr_L")
    write_dataset(out_o, orig["train"], orig["valid"], orig["test"],
                  ent2id, rel2id, patterns)
    write_dataset(out_l, train, valid, test, ent2id, rel2id, patterns)

    ent_tr_o = {e for h, _, t in orig["train"] for e in (h, t)}
    unseen = sum(1 for h, _, t in orig["test"]
                 if h not in ent_tr_o or t not in ent_tr_o)
    rep.update({"seed": args.seed, "split_sizes": list(sizes),
                "original_test_triples_with_entity_unseen_in_train": unseen,
                "relation_patterns": patterns})
    with open(os.path.join(out_l, "split_report.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=2)

    print("symmetric relations detected:",
          ", ".join(f"{r} {v:.3f}" for r, v in
                    sorted(rep["symmetric_relations"].items(),
                           key=lambda x: -x[1])))
    print("co-referent relation pairs detected:",
          rep["coreferent_relation_pairs"] or "none")
    print(f"groups: {rep['n_groups']}  sizes {rep['group_size_histogram']}")
    print(f"original test triples with an entity unseen in train: {unseen}")
    print(f"written: {out_o}  and  {out_l}  "
          f"(train/valid/test = {sizes[0]}/{sizes[1]}/{sizes[2]} each)")


if __name__ == "__main__":
    main()
