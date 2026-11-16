"""
ontology.py
-----------
Type assignment for every entity (derived from its name prefix) and
domain/range constraints for every relation, both used by the
type-constrained metrics that train_eval.py reports alongside the
unconstrained ones (the paper reports the tail-side type-constrained MRR).

Entity types
  UNIT        -- friendly or enemy fighting unit (BN/CO/PLT)
  COMMANDER   -- named commander entity
  ENGAGEMENT  -- engagement event
  MOVEMENT    -- movement event
  MISSION     -- planned mission
  LOCATION
  TERRAIN
  WEATHER
  EQUIPMENT
  TIME
"""

import re


def entity_type(name):
    if name.startswith("CDR_"):       return "COMMANDER"
    if name.startswith(("FR_", "EN_")):
        # FR_BN_*, FR_CO_*, FR_PLT_*, EN_BN_*, EN_CO_*, EN_PLT_*
        return "UNIT"
    if name.startswith("ENG_"):       return "ENGAGEMENT"
    if name.startswith("MOV_"):       return "MOVEMENT"
    if name.startswith("MSN_"):       return "MISSION"
    if name.startswith("LOC"):        return "LOCATION"
    if name.startswith("TER"):        return "TERRAIN"
    if name.startswith("WX"):         return "WEATHER"
    if name.startswith("EQ"):         return "EQUIPMENT"
    if re.match(r"^T_\d+$", name):    return "TIME"
    return "UNKNOWN"


# Domain (valid head types) and range (valid tail types) for every
# relation in the C2KG ontology, exactly as produced by the generator
# (data_gen.py); check_schema.py verifies every triple against this table.
# v1.1.1 corrects the domains of locatedAt, usesEquipment, movesTo,
# hasObstacle and occursAt, which in v1.1.0 did not match the generator.  The
# ranges were correct, so only head-side type-constrained metrics were
# affected.
RELATION_DOMAIN_RANGE = {
    # command hierarchy
    "belongsTo":       ({"UNIT"},                {"UNIT"}),
    "commands":        ({"COMMANDER"},           {"UNIT"}),
    # spatial / temporal context
    "locatedAt":       ({"UNIT", "ENGAGEMENT"},  {"LOCATION"}),
    "occursAt":        ({"ENGAGEMENT"},          {"LOCATION"}),
    "occursAtTime":    ({"ENGAGEMENT", "MOVEMENT"}, {"TIME"}),
    # event participation
    "hasParticipant":  ({"ENGAGEMENT"},          {"UNIT"}),
    "usesEquipment":   ({"UNIT", "ENGAGEMENT"},  {"EQUIPMENT"}),
    # mission / planning
    "hasMission":      ({"UNIT"},                {"MISSION"}),
    "hasDeadline":     ({"MISSION"},             {"TIME"}),
    # movement
    "movesTo":         ({"MOVEMENT"},            {"LOCATION"}),
    "hasOrigin":       ({"MOVEMENT"},            {"LOCATION"}),
    "executesMove":    ({"UNIT"},                {"MOVEMENT"}),
    # combat
    "defends":         ({"UNIT"},                {"LOCATION"}),
    "threatens":       ({"UNIT"},                {"UNIT"}),
    "threatenedBy":    ({"UNIT"},                {"UNIT"}),
    "supports":        ({"UNIT"},                {"UNIT"}),
    "supportedBy":     ({"UNIT"},                {"UNIT"}),
    "alliedWith":      ({"UNIT"},                {"UNIT"}),
    # terrain / environment
    "adjacentTo":      ({"LOCATION"},            {"LOCATION"}),
    "nearLocation":    ({"UNIT"},                {"LOCATION"}),
    "hasTerrain":      ({"LOCATION"},            {"TERRAIN"}),
    "hasWeather":      ({"LOCATION"},            {"WEATHER"}),
    "hasObstacle":     ({"UNIT"},                {"TERRAIN"}),
}


def build_type_mask(id2ent, rel2id):
    """For every relation, return two boolean numpy vectors
    (head_mask, tail_mask) of length n_ent indicating which entities
    are valid heads / tails under the ontology."""
    import numpy as np
    n_ent = len(id2ent)
    ent_type = [entity_type(id2ent[i]) for i in range(n_ent)]
    masks = {}
    for r, ri in rel2id.items():
        if r in RELATION_DOMAIN_RANGE:
            dom, rng = RELATION_DOMAIN_RANGE[r]
            hmask = np.array([t in dom for t in ent_type], dtype=bool)
            tmask = np.array([t in rng for t in ent_type], dtype=bool)
        else:
            hmask = np.ones(n_ent, dtype=bool)
            tmask = np.ones(n_ent, dtype=bool)
        masks[ri] = (hmask, tmask)
    return masks
