"""
competency_questions.py
-----------------------
Five competency questions over the whole graph (training, validation and
test triples) of a C2KG folder.  They show which C2 questions the schema
can express and how many answers the generator produces; they are not
evidence that the graph is realistic.

  CQ1  For a time T, which units have a mission due at T?
       Unit -hasMission-> Mission -hasDeadline-> T
  CQ2  Which friendly units are near the location of an enemy unit that
       threatens them?
       e -threatens-> f,  e -locatedAt-> l,  f -nearLocation-> l
  CQ3  For a unit U, which units support U and are near U's location?
       s -supports-> U,  U -locatedAt-> l,  s -nearLocation-> l
  CQ4  For a time T, a terrain type X and a weather condition Y, which
       engagements took place at T at a location with terrain X and
       weather Y?
       E -occursAtTime-> T,  E -occursAt-> l,  l -hasTerrain-> X,
       l -hasWeather-> Y
  CQ5  For an engagement E, which commander is responsible for each
       participating unit (the commander of a participating company, or of
       the company of a participating platoon)?
       E -hasParticipant-> p,  (p -belongsTo-> c, platoon) ,  k -commands-> c

For every question the script reports the number of answer tuples, the
number of parameter values with at least one answer and the number of
parameter values in the graph.

    python competency_questions.py --data ../data/c2kg_L
"""

import argparse
import os
from collections import defaultdict

from check_schema import load_graph
from ontology import entity_type


def answers(trip):
    out = defaultdict(lambda: defaultdict(set))
    for h, r, t in trip:
        out[r][h].add(t)
    ents = {e for h, _, t in trip for e in (h, t)}
    of = lambda ty: sorted(e for e in ents if entity_type(e) == ty)
    units, times, engs = of("UNIT"), of("TIME"), of("ENGAGEMENT")
    res = {}

    # CQ1: (T, unit)
    a = {(t, u) for u in units for m in out["hasMission"][u]
         for t in out["hasDeadline"][m]}
    res["CQ1"] = (len(a), len({t for t, _ in a}), len(times), "times")

    # CQ2: (e, f)
    a = {(e, f) for e, fs in out["threatens"].items() for f in fs
         if out["locatedAt"][e] & out["nearLocation"][f]}
    res["CQ2"] = (len(a), len({f for _, f in a}),
                  len({f for fs in out["threatens"].values() for f in fs}),
                  "threatened friendly units")

    # CQ3: (U, s)
    a = {(u, s) for s, us in out["supports"].items() for u in us
         if out["locatedAt"][u] & out["nearLocation"][s]}
    res["CQ3"] = (len(a), len({u for u, _ in a}),
                  len({u for us in out["supports"].values() for u in us}),
                  "supported units")

    # CQ4: ((T, X, Y), E)
    a = set()
    for e in engs:
        for t in out["occursAtTime"][e]:
            for l in out["occursAt"][e]:
                for x in out["hasTerrain"][l]:
                    for y in out["hasWeather"][l]:
                        a.add(((t, x, y), e))
    res["CQ4"] = (len(a), len({k for k, _ in a}),
                  len(times) * len(of("TERRAIN")) * len(of("WEATHER")),
                  "(time, terrain, weather) combinations")

    # CQ5: (E, k)
    cmd = defaultdict(set)
    for k, us in out["commands"].items():
        for u in us:
            cmd[u].add(k)
    a = set()
    for e in engs:
        for p in out["hasParticipant"][e]:
            cos = {p} if "_CO_" in p else out["belongsTo"][p]
            for c in cos:
                for k in cmd[c]:
                    a.add((e, k))
    res["CQ5"] = (len(a), len({e for e, _ in a}), len(engs), "engagements")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    trip = load_graph(args.data)
    print(f"graph of {os.path.basename(os.path.normpath(args.data))}: "
          f"{len(trip)} triples")
    for q, (n, k, m, what) in answers(trip).items():
        print(f"  {q}: {n:5d} answers; {k} of {m} {what} have an answer")


if __name__ == "__main__":
    main()
