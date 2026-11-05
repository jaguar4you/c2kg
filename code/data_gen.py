"""
data_gen.py
-----------
C2KG graph generator and the inverse-pair split (C2KG-N).

Operational C2 data cannot be released, so C2KG is a synthetic knowledge
graph built from an ontology organised around five of the METT-TC mission
variables (mission; enemy; terrain and weather; troops and support
available; time available) plus engagement and movement events.

The generator injects controlled relational regularities so that the
benchmark can be audited for leakage and analysed by relation pattern:

  * Symmetric relations      : adjacentTo, alliedWith
  * Antisymmetric relations  : belongsTo, commands, threatens, supports
  * Inverse relation pairs   : (threatens, threatenedBy), (supports, supportedBy)
  * Composition (2-hop rule) : locatedAt o adjacentTo -> nearLocation
  * N-to-N relations         : usesEquipment, hasParticipant

Usage
    python data_gen.py --out ../data/c2kg_N --seed 42 --split-mode grouped

Outputs (in --out dir):
  entities.dict, relations.dict, train.txt, valid.txt, test.txt,
  relation_patterns.json   (ground-truth relation -> pattern label)

The leakage-aware split (C2KG-L) is produced by split_fixed.py.
"""

import os, json, random, argparse
from collections import defaultdict

# ----------------------------------------------------------------------------
# Ontology-grounded vocabulary (METT-TC)
# ----------------------------------------------------------------------------
REL_PATTERNS = {
    "belongsTo":        "antisymmetric",   # company -> battalion (chain of command)
    "commands":         "antisymmetric",   # commander -> unit
    "locatedAt":        "general",         # unit/event -> location
    "occursAt":         "general",         # engagement -> location
    "occursAtTime":     "general",         # event -> time
    "hasParticipant":   "N-N",             # engagement -> unit
    "usesEquipment":    "N-N",             # unit/engagement -> equipment
    "hasMission":       "general",         # unit -> mission
    "hasDeadline":      "general",         # mission -> time
    "movesTo":          "general",         # movement -> location
    "hasOrigin":        "general",         # movement -> location
    "executesMove":     "antisymmetric",   # unit -> movement
    "defends":          "general",         # friendly unit -> location
    "threatens":        "antisymmetric",   # enemy unit -> friendly unit
    "threatenedBy":     "inverse",         # inverse of threatens
    "supports":         "antisymmetric",   # unit -> unit (fire/log support)
    "supportedBy":      "inverse",         # inverse of supports
    "alliedWith":       "symmetric",       # unit <-> unit
    "adjacentTo":       "symmetric",       # location <-> location
    "nearLocation":     "composition",     # locatedAt o adjacentTo
    "hasTerrain":       "general",         # location -> terrain
    "hasWeather":       "general",         # location -> weather
    "hasObstacle":      "general",         # unit -> terrain (mobility obstacle)
}
RELATIONS = list(REL_PATTERNS.keys())


def build_graph(seed=42,
                n_fr_bn=9, n_en_bn=9, comp_per_bn=4, plt_per_comp=3,
                n_locations=260, n_equipment=150, n_missions=90,
                n_time=36, n_terrain=14, n_weather=8,
                n_engagements=560, n_movements=440):
    rng = random.Random(seed)
    triples = set()

    def add(h, r, t):
        triples.add((h, r, t))

    # --- Friendly / Enemy order of battle (hierarchy) ----------------------
    fr_units, en_units, commanders = [], [], []

    def make_force(prefix, n_bn, container):
        for b in range(n_bn):
            bn = f"{prefix}_BN_{b}"
            container.append(bn)
            cdr = f"CDR_{bn}"
            commanders.append(cdr)
            add(cdr, "commands", bn)
            for c in range(comp_per_bn):
                co = f"{prefix}_CO_{b}_{c}"
                container.append(co)
                add(co, "belongsTo", bn)
                ccdr = f"CDR_{co}"
                commanders.append(ccdr)
                add(ccdr, "commands", co)
                for p in range(plt_per_comp):
                    pl = f"{prefix}_PLT_{b}_{c}_{p}"
                    container.append(pl)
                    add(pl, "belongsTo", co)
        return container

    make_force("FR", n_fr_bn, fr_units)
    make_force("EN", n_en_bn, en_units)

    locations  = [f"LOC_{i}"  for i in range(n_locations)]
    equipment  = [f"EQ_{i}"   for i in range(n_equipment)]
    missions   = [f"MSN_{i}"  for i in range(n_missions)]
    times      = [f"T_{i}"    for i in range(n_time)]
    terrains   = [f"TER_{i}"  for i in range(n_terrain)]
    weathers   = [f"WX_{i}"   for i in range(n_weather)]
    engagements= [f"ENG_{i}"  for i in range(n_engagements)]
    movements  = [f"MOV_{i}"  for i in range(n_movements)]

    # --- Location topology: adjacency (symmetric) + terrain/weather --------
    for i, loc in enumerate(locations):
        # ring + random chords -> connected adjacency graph
        nbrs = {locations[(i + 1) % n_locations]}
        for _ in range(rng.randint(1, 3)):
            nbrs.add(rng.choice(locations))
        nbrs.discard(loc)
        for nb in nbrs:
            add(loc, "adjacentTo", nb)
            add(nb, "adjacentTo", loc)               # enforce symmetry
        add(loc, "hasTerrain", rng.choice(terrains))
        add(loc, "hasWeather", rng.choice(weathers))

    adj = defaultdict(set)
    for (h, r, t) in triples:
        if r == "adjacentTo":
            adj[h].add(t)

    # --- Unit deployment / equipment / missions ---------------------------
    all_units = fr_units + en_units
    for u in all_units:
        loc = rng.choice(locations)
        add(u, "locatedAt", loc)
        # composition rule: locatedAt o adjacentTo -> nearLocation
        for nb in adj[loc]:
            add(u, "nearLocation", nb)
        # N-to-N equipment
        for eq in rng.sample(equipment, rng.randint(1, 4)):
            add(u, "usesEquipment", eq)
        if u.endswith(tuple(f"BN_{i}" for i in range(20))) or "CO_" in u:
            msn = rng.choice(missions)
            add(u, "hasMission", msn)
            add(msn, "hasDeadline", rng.choice(times))
        # mobility obstacle (unit experiences a terrain obstacle)
        if rng.random() < 0.4:
            add(u, "hasObstacle", rng.choice(terrains))

    # --- alliedWith (symmetric, within same force) ------------------------
    for force in (fr_units, en_units):
        for _ in range(len(force)):
            a, b = rng.sample(force, 2)
            add(a, "alliedWith", b)
            add(b, "alliedWith", a)

    # --- supports / supportedBy (antisymmetric + inverse) -----------------
    for _ in range(int(0.8 * len(all_units))):
        force = fr_units if rng.random() < 0.5 else en_units
        a, b = rng.sample(force, 2)
        add(a, "supports", b)
        add(b, "supportedBy", a)                     # inverse rule

    # --- threatens / threatenedBy (antisymmetric + inverse, cross-force) --
    for _ in range(int(1.2 * n_fr_bn * comp_per_bn)):
        e = rng.choice(en_units)
        f = rng.choice(fr_units)
        add(e, "threatens", f)
        add(f, "threatenedBy", e)                    # inverse rule

    # --- defends (friendly units defend locations) ------------------------
    for f in fr_units:
        if rng.random() < 0.5:
            add(f, "defends", rng.choice(locations))

    # --- Engagements (events): the operational core -----------------------
    for eng in engagements:
        loc = rng.choice(locations)
        add(eng, "occursAt", loc)
        add(eng, "locatedAt", loc)
        add(eng, "occursAtTime", rng.choice(times))
        fr = rng.choice([u for u in fr_units if "PLT" in u or "CO_" in u])
        en = rng.choice([u for u in en_units if "PLT" in u or "CO_" in u])
        add(eng, "hasParticipant", fr)               # N-to-N
        add(eng, "hasParticipant", en)               # N-to-N
        for eq in rng.sample(equipment, rng.randint(1, 3)):
            add(eng, "usesEquipment", eq)

    # --- Movements (events) ----------------------------------------------
    for mov in movements:
        org, dst = rng.sample(locations, 2)
        add(mov, "hasOrigin", org)
        add(mov, "movesTo", dst)
        add(mov, "occursAtTime", rng.choice(times))
        u = rng.choice(all_units)
        add(u, "executesMove", mov)

    return sorted(triples), RELATIONS


def split_triples(triples, seed=42, valid_ratio=0.10, test_ratio=0.10):
    """
    Random edge holdout with a coverage guarantee: every entity and every
    relation must appear at least once in the training graph (transductive
    setting required by the graph encoders).
    """
    rng = random.Random(seed)
    triples = list(triples)
    rng.shuffle(triples)

    ent_count = defaultdict(int)
    rel_count = defaultdict(int)
    for h, r, t in triples:
        ent_count[h] += 1
        ent_count[t] += 1
        rel_count[r] += 1

    n = len(triples)
    n_test = int(test_ratio * n)
    n_valid = int(valid_ratio * n)

    test, valid, train = [], [], []
    ec, rc = defaultdict(int), defaultdict(int)

    def safe_to_hold(h, r, t):
        return (ent_count[h] - ec[h] > 1 and
                ent_count[t] - ec[t] > 1 and
                rel_count[r] - rc[r] > 1)

    for (h, r, t) in triples:
        if len(test) < n_test and safe_to_hold(h, r, t):
            test.append((h, r, t)); ec[h]+=1; ec[t]+=1; rc[r]+=1
        elif len(valid) < n_valid and safe_to_hold(h, r, t):
            valid.append((h, r, t)); ec[h]+=1; ec[t]+=1; rc[r]+=1
        else:
            train.append((h, r, t))
    return train, valid, test


INVERSE_PAIRS = (("threatens", "threatenedBy"),
                 ("supports",  "supportedBy"))


def split_triples_grouped(triples, seed=42, valid_ratio=0.10, test_ratio=0.10,
                          inverse_pairs=INVERSE_PAIRS):
    """
    Inverse-pair grouped split.  For every inverse-relation pair (r1, r2),
    a triple (h, r1, t) and its inverse counterpart (t, r2, h) are placed in
    the same split atomically; this removes the trivial leakage in which a
    held-out (t, r2, h) is solved by memorising the still-present
    (h, r1, t) in training.  All other triples form singleton groups.

    Coverage guarantee (every entity and every relation appears in train)
    is enforced at the group level.
    """
    rng = random.Random(seed)
    triples = list(triples)
    triple_set = set(triples)
    inv = {a: b for a, b in inverse_pairs}
    inv.update({b: a for a, b in inverse_pairs})

    # build atomic groups (size 1 or 2)
    visited, groups = set(), []
    for h, r, t in triples:
        if (h, r, t) in visited:
            continue
        grp = [(h, r, t)]; visited.add((h, r, t))
        if r in inv:
            mate = (t, inv[r], h)
            if mate in triple_set and mate not in visited:
                grp.append(mate); visited.add(mate)
        groups.append(grp)
    rng.shuffle(groups)

    ent_count = defaultdict(int); rel_count = defaultdict(int)
    for h, r, t in triples:
        ent_count[h] += 1; ent_count[t] += 1; rel_count[r] += 1

    n = len(triples)
    n_test = int(test_ratio * n); n_valid = int(valid_ratio * n)
    test, valid, train = [], [], []
    ec, rc = defaultdict(int), defaultdict(int)

    def safe(grp):
        eh, rh = defaultdict(int), defaultdict(int)
        for h, r, t in grp:
            eh[h] += 1; eh[t] += 1; rh[r] += 1
        return (all(ent_count[e] - ec[e] - eh[e] > 0 for e in eh) and
                all(rel_count[r_] - rc[r_] - rh[r_] > 0 for r_ in rh))

    def push(dst, grp):
        dst.extend(grp)
        for h, r, t in grp:
            ec[h] += 1; ec[t] += 1; rc[r] += 1

    for grp in groups:
        gsz = len(grp)
        if len(test) + gsz <= n_test and safe(grp):
            push(test, grp)
        elif len(valid) + gsz <= n_valid and safe(grp):
            push(valid, grp)
        else:
            push(train, grp)
    return train, valid, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["random", "grouped"],
                    default="grouped",
                    help="random: i.i.d. edge holdout (inverse-pair "
                         "leakage possible). grouped: inverse pairs "
                         "co-assigned to the same split (no inverse "
                         "leakage).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    triples, relations = build_graph(seed=args.seed)
    if args.split_mode == "grouped":
        train, valid, test = split_triples_grouped(triples, seed=args.seed)
    else:
        train, valid, test = split_triples(triples, seed=args.seed)

    entities = sorted({e for (h, r, t) in triples for e in (h, t)})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    def dump(name, rows):
        with open(os.path.join(args.out, name), "w") as f:
            for h, r, t in rows:
                f.write(f"{h}\t{r}\t{t}\n")

    dump("train.txt", train)
    dump("valid.txt", valid)
    dump("test.txt",  test)

    with open(os.path.join(args.out, "entities.dict"), "w") as f:
        for e, i in ent2id.items():
            f.write(f"{i}\t{e}\n")
    with open(os.path.join(args.out, "relations.dict"), "w") as f:
        for r, i in rel2id.items():
            f.write(f"{i}\t{r}\n")
    with open(os.path.join(args.out, "relation_patterns.json"), "w") as f:
        json.dump(REL_PATTERNS, f, indent=2)

    print(f"Entities : {len(entities)}")
    print(f"Relations: {len(relations)}")
    print(f"Triples  : total={len(triples)}  "
          f"train={len(train)}  valid={len(valid)}  test={len(test)}")
    deg = defaultdict(int)
    for h, r, t in triples:
        deg[h] += 1; deg[t] += 1
    print(f"Avg degree: {2*len(triples)/len(entities):.2f}")


if __name__ == "__main__":
    main()
