"""
check_schema.py
---------------
Automated checks of the C2KG graphs against the schema and the generation
rules of data_gen.py.  Every check is run on the whole graph (training,
validation and test triples, plus filter_extra.txt of a leak-blocked
variant), so the result does not depend on the split.

Hard constraints (the script fails if one is violated)
  typing         every entity has a type; every triple satisfies the domain
                 and range of its relation (ontology.py)
  symmetric      adjacentTo, alliedWith: (h, r, t) implies (t, r, h); no
                 self-loops
  antisymmetric  belongsTo, supports, threatens (same entity type at both
                 ends) and commands, executesMove (different types): no pair
                 in both directions
  inverse        supports(a, b) iff supportedBy(b, a); threatens(a, b) iff
                 threatenedBy(b, a)
  composition    for every unit u: nearLocation(u, z) iff there is y with
                 locatedAt(u, y) and adjacentTo(y, z)
  co-reference   for every engagement e: occursAt(e, l) iff locatedAt(e, l)
  hierarchy      every platoon belongs to exactly one company and every
                 company to exactly one battalion of the same force and with
                 the matching index; battalions belong to nothing; every
                 battalion and company has exactly one commander and every
                 commander commands exactly one unit
  force          supports and alliedWith stay within one force; threatens
                 goes from an enemy to a friendly unit; defends has friendly
                 heads; every engagement has exactly one friendly and one
                 enemy participant (company or platoon)
  functional     every unit and engagement is located at exactly one
                 location; every engagement and movement has exactly one
                 time; every movement has one origin, one destination
                 different from it and one executing unit; every location
                 has exactly one terrain and one weather
  connectivity   the adjacentTo graph is connected

Reported, not enforced (cardinalities that the generator leaves free)
  missions per unit, deadlines per mission, units per mission, equipment
  per unit and per engagement, obstacles per unit, defended locations.
  The generator draws one deadline per mission assignment, so a mission
  shared by k units can have up to k deadlines; times are nominal labels
  without an order.

Usage
    python check_schema.py                  # every C2KG folder in ../data
    python check_schema.py --data ../data/c2kg_L
"""

import argparse
import os
import re
from collections import Counter, defaultdict

from ontology import RELATION_DOMAIN_RANGE, entity_type

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FILES = ("train.txt", "valid.txt", "test.txt", "filter_extra.txt")


def load_graph(folder):
    trip = set()
    for f in FILES:
        p = os.path.join(folder, f)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if line:
                    trip.add(tuple(line.split("\t")))
    return trip


def force(u):
    return u.split("_")[0]          # FR or EN


def check(trip):
    """Return a list of (name, n_checked, n_violations, note)."""
    by = defaultdict(set)
    for h, r, t in trip:
        by[r].add((h, t))
    out_ = defaultdict(lambda: defaultdict(set))
    for h, r, t in trip:
        out_[r][h].add(t)
    ents = {e for h, _, t in trip for e in (h, t)}
    etype = {e: entity_type(e) for e in ents}
    res = []

    # typing
    bad_e = [e for e in ents if etype[e] == "UNKNOWN"]
    bad_t = [x for x in trip if x[1] not in RELATION_DOMAIN_RANGE
             or etype[x[0]] not in RELATION_DOMAIN_RANGE[x[1]][0]
             or etype[x[2]] not in RELATION_DOMAIN_RANGE[x[1]][1]]
    res.append(("typing: entities", len(ents), len(bad_e), ""))
    res.append(("typing: domain and range", len(trip), len(bad_t), ""))

    # symmetric
    for r in ("adjacentTo", "alliedWith"):
        p = by[r]
        v = sum(1 for (h, t) in p if (t, h) not in p or h == t)
        res.append((f"symmetric: {r}", len(p), v, ""))

    # antisymmetric
    for r, kind in (("belongsTo", "same type"), ("supports", "same type"),
                    ("threatens", "same type"), ("commands", "type-separated"),
                    ("executesMove", "type-separated")):
        p = by[r]
        v = sum(1 for (h, t) in p if (t, h) in p)
        res.append((f"antisymmetric: {r}", len(p), v, kind))

    # inverse
    for a, b in (("supports", "supportedBy"), ("threatens", "threatenedBy")):
        pa, pb = by[a], {(t, h) for (h, t) in by[b]}
        res.append((f"inverse: {a} / {b}", len(pa | pb), len(pa ^ pb), ""))

    # composition
    units = [e for e in ents if etype[e] == "UNIT"]
    v = 0
    for u in units:
        derived = {z for y in out_["locatedAt"][u] for z in out_["adjacentTo"][y]}
        v += len(derived ^ out_["nearLocation"][u])
    res.append(("composition: locatedAt o adjacentTo = nearLocation",
                len(by["nearLocation"]), v, "units"))

    # co-reference
    engs = [e for e in ents if etype[e] == "ENGAGEMENT"]
    v = sum(len(out_["occursAt"][e] ^ out_["locatedAt"][e]) for e in engs)
    res.append(("co-reference: occursAt = locatedAt", len(engs), v,
                "engagements"))

    # hierarchy
    parent = {h: out_["belongsTo"][h] for h in out_["belongsTo"]}
    v = 0
    n = 0
    for u in units:
        n += 1
        m = re.match(r"^(FR|EN)_(BN|CO|PLT)_(\d+)(?:_(\d+))?(?:_(\d+))?$", u)
        if not m:
            v += 1
            continue
        f, lvl, b, c, _ = m.groups()
        ps = parent.get(u, set())
        if lvl == "BN":
            v += len(ps) != 0
        elif lvl == "CO":
            v += ps != {f"{f}_BN_{b}"}
        else:
            v += ps != {f"{f}_CO_{b}_{c}"}
    res.append(("hierarchy: unit -> parent", n, v, "platoon -> company -> battalion"))
    cmd_units = Counter(t for (h, t) in by["commands"])
    cmd_heads = Counter(h for (h, t) in by["commands"])
    commandable = [u for u in units if "_PLT_" not in u]
    v = sum(cmd_units.get(u, 0) != 1 for u in commandable) + \
        sum(1 for c in cmd_heads.values() if c != 1) + \
        sum(1 for (h, t) in by["commands"] if h != f"CDR_{t}")
    res.append(("hierarchy: one commander per battalion and company",
                len(commandable), v, ""))

    # force
    v = sum(1 for r in ("supports", "alliedWith") for (h, t) in by[r]
            if force(h) != force(t))
    res.append(("force: supports, alliedWith within one force",
                len(by["supports"]) + len(by["alliedWith"]), v, ""))
    v = sum(1 for (h, t) in by["threatens"] if not (force(h) == "EN" and force(t) == "FR"))
    res.append(("force: threatens enemy -> friendly", len(by["threatens"]), v, ""))
    v = sum(1 for (h, t) in by["defends"] if force(h) != "FR")
    res.append(("force: defends by friendly units", len(by["defends"]), v, ""))
    v = 0
    for e in engs:
        ps = out_["hasParticipant"][e]
        fs = sorted(force(p) for p in ps)
        ok = fs == ["EN", "FR"] and all("_BN_" not in p for p in ps)
        v += not ok
    res.append(("force: one friendly and one enemy participant", len(engs), v,
                "company or platoon"))

    # functional
    movs = [e for e in ents if etype[e] == "MOVEMENT"]
    locs = [e for e in ents if etype[e] == "LOCATION"]
    def one(rel, heads):
        return sum(len(out_[rel][h]) != 1 for h in heads)
    res.append(("functional: locatedAt (units, engagements)",
                len(units) + len(engs), one("locatedAt", units + engs), ""))
    res.append(("functional: occursAtTime (engagements, movements)",
                len(engs) + len(movs), one("occursAtTime", engs + movs), ""))
    executors = Counter(t for (h, t) in by["executesMove"])
    v = one("hasOrigin", movs) + one("movesTo", movs) + \
        sum(executors.get(m, 0) != 1 for m in movs) + \
        sum(1 for m in movs if out_["hasOrigin"][m] == out_["movesTo"][m])
    res.append(("functional: movement origin, destination, executor",
                len(movs), v, "destination differs from origin"))
    res.append(("functional: hasTerrain, hasWeather (locations)", len(locs),
                one("hasTerrain", locs) + one("hasWeather", locs), ""))

    # connectivity
    adj = defaultdict(set)
    for h, t in by["adjacentTo"]:
        adj[h].add(t)
    seen, stack = set(), [locs[0]] if locs else []
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(adj[x] - seen)
    res.append(("connectivity: adjacentTo", len(locs), len(locs) - len(seen), ""))

    # free cardinalities
    free = {}
    def dist(rel, heads=None, inverse=False):
        if inverse:
            c = Counter(t for (h, t) in by[rel])
            return Counter(c.values())
        return Counter(len(out_[rel][h]) for h in heads)
    missions = [e for e in ents if etype[e] == "MISSION"]
    free["missions per battalion/company"] = dist("hasMission", commandable)
    free["deadlines per assigned mission"] = Counter(
        len(out_["hasDeadline"][m]) for m in missions if out_["hasDeadline"][m])
    free["units per assigned mission"] = dist("hasMission", inverse=True)
    free["equipment per unit"] = dist("usesEquipment", units)
    free["equipment per engagement"] = dist("usesEquipment", engs)
    free["obstacles per unit"] = dist("hasObstacle", units)
    free["defended locations per friendly unit"] = dist(
        "defends", [u for u in units if force(u) == "FR"])
    return res, free


def report(folder):
    trip = load_graph(folder)
    res, free = check(trip)
    print(f"graph of {os.path.basename(folder)}: {len(trip)} triples")
    fails = 0
    for name, n, v, note in res:
        fails += v > 0
        print(f"  {'PASS' if v == 0 else 'FAIL'}  {name:55s} "
              f"checked {n:5d}  violations {v}{'  (' + note + ')' if note else ''}")
    print("  free cardinalities (value: count):")
    for k, c in free.items():
        print(f"    {k:40s} " + ", ".join(f"{a}: {b}" for a, b in sorted(c.items())))
    return fails, trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=[])
    args = ap.parse_args()
    folders = args.data or [os.path.join(DATA, d) for d in sorted(os.listdir(DATA))
                            if d.startswith("c2kg_")
                            and os.path.exists(os.path.join(DATA, d, "train.txt"))]
    total, graphs = 0, {}
    for f in folders:
        fails, trip = report(f)
        total += fails
        graphs[os.path.basename(os.path.normpath(f))] = trip
        print()
    # splits of one graph seed must hold the same graph
    seeds = defaultdict(list)
    for name in graphs:
        m = re.match(r"^c2kg_(?:L|N)(?:_blocked)?(?:_g(\d+))?$", name)
        if m:
            seeds[m.group(1) or "42"].append(name)
    for s, names in sorted(seeds.items()):
        same = all(graphs[n] == graphs[names[0]] for n in names)
        total += not same
        print(f"{'PASS' if same else 'FAIL'}  graph seed {s}: "
              f"{', '.join(sorted(names))} hold the same {len(graphs[names[0]])} triples")
    if total:
        raise SystemExit(f"\n{total} check(s) failed")
    print("\nall schema checks passed")


if __name__ == "__main__":
    main()
