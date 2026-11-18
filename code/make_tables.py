"""
make_tables.py
--------------
Computes every number reported in the paper from the released result files
and writes the tables.  Nothing is trained; the script only reads

  ../results/runs/*.json.gz       trained models (unpacked or not)
  ../results/probe_*.json         training-free baselines (path_probe.py)
  ../data/                        the datasets (patterns, leak flags)

and writes

  ../results/paper/numbers.txt    every number quoted in the text
  ../results/paper/*.tex          the tables of the paper (LaTeX)

    python make_tables.py

Conventions.  "Tail MRR" is the filtered tail-query MRR (exact expectation
under random tie-breaking); "TC" denotes the tail-side type-constrained
metric.  The selected protocol is stage 2 (stage 1 where no stage 2 was
selected); WS-RGCN is the warm-started encoder.  Means and sample standard
deviations are over training seeds; tests are paired over seeds.
Bootstrap intervals resample test triples (10,000 resamples, fixed seed).
"""

import gzip
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from ontology import entity_type
from make_blocked import partners

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
RUNS = os.path.join(RES, "runs")
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
OUT = os.path.join(RES, "paper")

MODELS = ["transe", "rotate", "distmult", "complex", "rgcn", "compgcn"]
M7 = MODELS + ["re_rgcn"]
NAME = {"transe": "TransE", "rotate": "RotatE", "distmult": "DistMult",
        "complex": "ComplEx", "rgcn": "R-GCN", "compgcn": "CompGCN",
        "re_rgcn": "WS-RGCN"}
ORDER = ["rotate", "re_rgcn", "complex", "distmult", "transe", "compgcn",
         "rgcn"]                                   # row order of the tables
PATS = ["symmetric", "antisymmetric", "inverse", "composition", "N-N",
        "general"]
SAME_TYPE_ANTI = {"belongsTo", "supports", "threatens"}
SHARED = 1e-2
MARGIN = 0.01                                      # TOST equivalence margin

_cache = {}
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def load(name):
    if name not in _cache:
        p = os.path.join(RUNS, name + ".json.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt") as f:
                _cache[name] = json.load(f)
        else:
            _cache[name] = json.load(open(os.path.join(RUNS, name + ".json")))
    return _cache[name]


def probe(name):
    return json.load(open(os.path.join(RES, f"probe_{name}.json")))


def lstr(x):
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def runs(R, prefix):
    return {k.rsplit("|", 1)[1]: v for k, v in R.items()
            if "|" in k and k.startswith(prefix)}


def prot(R, model, which):
    """Runs of one protocol, keyed by seed ('s0' ...)."""
    if model == "re_rgcn":
        w = R["selected"]["re_rgcn"]
        return runs(R, f"ws|re_rgcn|lr1{lstr(w['lr1'])}|emb{lstr(w['emb'])}"
                       f"|lr{lstr(w['gnn'])}|")
    s = R["selected"][model]
    if which == "shared":
        return runs(R, f"s1|{model}|lr{lstr(SHARED)}|")
    if which == "s1" or (which == "s2" and s["lr2"] is None):
        return runs(R, f"s1|{model}|lr{lstr(s['lr1'])}|")
    return runs(R, f"s2|{model}|lr1{lstr(s['lr1'])}|lr{lstr(s['lr2'])}|")


GET = {"tail": lambda r: r["test"]["tail_only"]["MRR"],
       "ht": lambda r: r["test"]["MRR"],
       "h1": lambda r: r["test"]["tail_only"]["H@1"],
       "h3": lambda r: r["test"]["tail_only"]["H@3"],
       "h10": lambda r: r["test"]["tail_only"]["H@10"],
       "tc": lambda r: r["test"]["constrained"]["tail_only"]["MRR"],
       "ht_h1": lambda r: r["test"]["H@1"],
       "ht_h10": lambda r: r["test"]["H@10"],
       "val": lambda r: r["val_mrr"],
       "best": lambda r: r["best_epoch"],
       "gnn": lambda r: r.get("gnn_change", np.nan)}


def v(d, key="tail"):
    """Per-seed values (sorted by seed) of a metric."""
    return np.array([GET[key](d[k]) for k in sorted(d)])


def ms(a, nd=4):
    a = np.asarray(a, float)
    return (f"{a.mean():.{nd}f} $\\pm$ {a.std(ddof=1):.{nd}f}" if len(a) > 1
            else f"{a.mean():.{nd}f}")


_H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, 60001))])


def rr(g, m):
    g = np.asarray(g, int)
    m = np.asarray(m, int)
    return (_H[g + m] - _H[g]) / m


def per_query(d, side="tail", constrained=False):
    """Seeds x test triples matrix of reciprocal ranks."""
    out = []
    for k in sorted(d):
        p = d[k]["test"]["per_triple"]
        if side == "tail":
            out.append(rr(p["tail_g_c"], p["tail_m_c"]) if constrained
                       else rr(p["tail_g"], p["tail_m"]))
        else:
            out.append(rr(p["head_g"], p["head_m"]))
    return np.array(out)


def probe_rr(name, which):
    p = probe(name)[which]["per_triple"]
    return rr(p["tail_g"], p["tail_m"])


def triples(folder, split="test"):
    with open(os.path.join(DATA, folder, f"{split}.txt"), encoding="utf-8") as f:
        return [tuple(l.rstrip("\r\n").split("\t")) for l in f if l.strip()]


def patterns_of(folder):
    pat = json.load(open(os.path.join(DATA, folder, "relation_patterns.json")))
    te = triples(folder)
    return (np.array([r for _, r, _ in te]),
            np.array([pat[r] for _, r, _ in te]))


def tost(d, margin=MARGIN, alpha=0.05):
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p2 = stats.t.cdf((m - margin) / se, n - 1)
    tc = stats.t.ppf(1 - alpha, n - 1)
    return max(p1, p2), (m - tc * se, m + tc * se)


def ci95(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return d.mean() - tc * se, d.mean() + tc * se


def paired(a, b):
    """mean and sd of a-b and two-sided paired t-test p."""
    d = np.asarray(a) - np.asarray(b)
    return d.mean(), d.std(ddof=1), stats.ttest_rel(a, b).pvalue


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def sgn(x, nd=4):
    return f"${'+' if x >= 0 else '-'}${abs(x):.{nd}f}"


def pfmt(p):
    if p >= 0.001:
        return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def fmt_lr(x):
    if x is None:
        return "--"
    e = int(np.floor(np.log10(x) + 1e-9))
    m = round(x / 10 ** e)
    if e == 0:
        return f"{m}"
    return (f"$10^{{{e}}}$" if m == 1 else f"$3{{\\times}}10^{{{e}}}$")


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", newline="\n") as f:
        f.write(text)


# ------------------------------------------------------------------------
def section_checks():
    say("== 0. Result files")
    n_runs = 0
    gpu = 0.0
    tied = []
    nan = []
    tmax = {}
    for f in sorted(os.listdir(RUNS)):
        name = f.split(".json")[0]
        R = load(name)
        meta = R["_meta"]
        rs = [x for k, x in R.items() if "|" in k]
        n_runs += len(rs)
        gpu += sum(x.get("time", 0.0) for x in rs)
        tied += [x["val_tied_frac"] for x in rs]
        grp = "wn" if name.startswith("wn") else "c2kg"
        tmax[grp] = max(tmax.get(grp, 0.0), max(x["val_tied_frac"] for x in rs))
        nan += [x["val_nan_frac"] for x in rs]
        say(f"  {name:16s} runs {len(rs):4d}  data {meta['data_sha'][:12]}  "
            f"code {meta['code_sha'][:12]}  bs {meta['bs']}  d {meta['dim']}"
            f"  margin {meta['margin']}  epochs {meta['epochs']}")
    say(f"  total training runs {n_runs}; GPU time {gpu / 3600:.1f} h; "
        f"largest validation tie fraction {max(tied):.4f}; "
        f"largest NaN fraction {max(nan):.4f}; largest tie fraction C2KG "
        f"{tmax['c2kg']:.4f}, WN18RR {tmax['wn']:.4f}")


def section_reference():
    say("\n== 1. Reference results, C2KG-L (graph 42)")
    R = load("c2kg_L")
    rows = []
    for m in ORDER:
        d = prot(R, m, "s2")
        rows.append((NAME[m], ms(v(d)), f"{v(d, 'h1').mean():.3f}",
                     f"{v(d, 'h3').mean():.3f}", f"{v(d, 'h10').mean():.3f}",
                     ms(v(d, "ht")), f"{v(d, 'tc').mean():.4f}"))
        say(f"  {NAME[m]:9s} tail {ms(v(d))}  H@1 {v(d, 'h1').mean():.3f} "
            f"H@3 {v(d, 'h3').mean():.3f} H@10 {v(d, 'h10').mean():.3f}  "
            f"h+t {ms(v(d, 'ht'))}  TC {v(d, 'tc').mean():.4f}  "
            f"n={len(d)}")
    P = probe("L")
    base = []
    for key, nm in (("rule", "Rule miner"), ("path", "Path rule"),
                    ("type", "Type-constrained random"),
                    ("freq", "Frequency")):
        o = P[key]["overall"]
        base.append((nm, f"{o['MRR']:.4f}", f"{o['H@1']:.3f}", "--",
                     f"{o['H@10']:.3f}", "--",
                     f"{o['MRR']:.4f}" if key == "type" else "--"))
        say(f"  {nm:24s} tail {o['MRR']:.4f} H@1 {o['H@1']:.4f} "
            f"H@10 {o['H@10']:.4f}")
    say(f"  rule selected {P['rule']['selected']}")
    body = "\n".join(" & ".join(r) + r"\\" for r in rows)
    body += "\n\\midrule\n" + "\n".join(" & ".join(r) + r"\\" for r in base)
    write("tab_reference", body + "\n")


def section_patterns():
    say("\n== 2. Patterns (TC tail MRR), C2KG-L")
    R = load("c2kg_L")
    rel, pat = patterns_of("c2kg_L")
    groups = [("Sym.", pat == "symmetric"),
              ("Anti. same", (pat == "antisymmetric")
               & np.isin(rel, list(SAME_TYPE_ANTI))),
              ("Anti. typed", (pat == "antisymmetric")
               & ~np.isin(rel, list(SAME_TYPE_ANTI))),
              ("Inv.", pat == "inverse"), ("Comp.", pat == "composition"),
              ("N-N", pat == "N-N"), ("Gen.", pat == "general")]
    say("  n " + " ".join(f"{g}={int(mask.sum())}" for g, mask in groups))
    rows = [["Test triples"] + [str(int(mask.sum())) for _, mask in groups]]
    for m in ORDER:
        Q = per_query(prot(R, m, "s2"), constrained=True)
        vals = [Q[:, mask].mean() for _, mask in groups]
        rows.append([NAME[m]] + [f"{x:.3f}" for x in vals])
        say(f"  {NAME[m]:9s} " + " ".join(f"{x:.3f}" for x in vals))
    for key, nm in (("type", "Type-constrained random"),
                    ("rule", "Rule miner"), ("freq", "Frequency")):
        q = probe_rr("L", key)
        vals = [q[mask].mean() for _, mask in groups]
        rows.append([nm] + [f"{x:.3f}" for x in vals])
        say(f"  {nm:24s} " + " ".join(f"{x:.3f}" for x in vals))
    body = " & ".join(rows[0]) + r"\\" + "\n\\midrule\n"
    body += "\n".join(" & ".join(r) + r"\\" for r in rows[1:8])
    body += "\n\\midrule\n" + "\n".join(" & ".join(r) + r"\\" for r in rows[8:])
    write("tab_patterns", body + "\n")

    say("\n== 2b. Relations (TC tail MRR), C2KG-L")
    rels = sorted(set(rel), key=lambda r: (PATS.index(pat[rel == r][0]), r))
    Qs = {m: per_query(prot(R, m, "s2"), constrained=True) for m in ORDER}
    tq = probe_rr("L", "type")
    rq = probe_rr("L", "rule")
    rows = []
    learn = 0
    for r in rels:
        mask = rel == r
        vals = [Qs[m][:, mask].mean() for m in ORDER]
        rows.append([f"\\texttt{{{r}}}", str(int(mask.sum()))]
                    + [f"{x:.3f}" for x in vals]
                    + [f"{tq[mask].mean():.3f}", f"{rq[mask].mean():.3f}"])
        say(f"  {r:15s} n={int(mask.sum()):3d} "
            + " ".join(f"{x:.3f}" for x in vals)
            + f" | type {tq[mask].mean():.3f} rule {rq[mask].mean():.3f}")
    write("tab_relations", "\n".join(" & ".join(x) + r"\\" for x in rows) + "\n")
    # share of the test set in relations linked to other facts
    te = triples("c2kg_L")
    linked = np.array([r in ("adjacentTo", "nearLocation")
                       or (r == "locatedAt" and entity_type(h) == "UNIT")
                       for h, r, t in te])
    say(f"  test triples of adjacentTo, nearLocation and unit locatedAt: "
        f"{linked.sum()} of {len(te)} ({100 * linked.mean():.1f}%)")
    for m in ("rotate", "complex"):
        Q = Qs[m]
        say(f"  {NAME[m]}: TC MRR on these {Q[:, linked].mean():.3f}, on the "
            f"rest {Q[:, ~linked].mean():.3f}; type-random rest "
            f"{tq[~linked].mean():.3f}")
    say(f"  rule miner: on these {rq[linked].mean():.3f}, on the rest "
        f"{rq[~linked].mean():.3f}")


def section_leakage():
    say("\n== 3. Leakage: C2KG-L, C2KG-N, C2KG-N-blocked (tail MRR)")
    L, N, B = load("c2kg_L"), load("c2kg_N"), load("c2kg_Nb")
    rows = []
    rank = {}
    for split, R in (("L", L), ("N", N), ("Nb", B)):
        rank[split] = sorted(M7, key=lambda m: -v(prot(R, m, "s2")).mean())
    for m in ORDER:
        a, b, c = (v(prot(R, m, "s2")) for R in (L, N, B))
        rows.append([NAME[m], ms(a), ms(b), ms(c),
                     f"{b.mean() / a.mean():.1f}",
                     f"{rank['L'].index(m) + 1} / {rank['N'].index(m) + 1}"])
        say(f"  {NAME[m]:9s} L {ms(a)}  N {ms(b)}  Nb {ms(c)}  "
            f"N/L {b.mean() / a.mean():.2f}  rank L {rank['L'].index(m) + 1} "
            f"N {rank['N'].index(m) + 1}")
    for key, nm in (("rule", "Rule miner"), ("path", "Path rule"),
                    ("type", "Type-constrained random"), ("freq", "Frequency")):
        a, b, c = (probe(s)[key]["overall"]["MRR"]
                   for s in ("L", "N", "N_blocked"))
        rows.append([nm, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}",
                     f"{b / a:.1f}", "--"])
        say(f"  {nm:24s} L {a:.4f} N {b:.4f} Nb {c:.4f}")
    say("  order L: " + ", ".join(NAME[m] for m in rank["L"]))
    say("  order N: " + ", ".join(NAME[m] for m in rank["N"]))
    sh = {s: [v(prot(R, m, "shared")).mean() for m in MODELS]
          for s, R in (("L", L), ("N", N))}
    say("  shared rate L: " + " ".join(f"{NAME[m]} {x:.4f}"
                                       for m, x in zip(MODELS, sh["L"])))
    say("  shared rate N: " + " ".join(f"{NAME[m]} {x:.4f}"
                                       for m, x in zip(MODELS, sh["N"])))
    body = "\n".join(" & ".join(r) + r"\\" for r in rows[:7])
    body += "\n\\midrule\n" + "\n".join(" & ".join(r) + r"\\" for r in rows[7:])
    write("tab_leakage_effect", body + "\n")

    say("\n== 3b. Same test triples: C2KG-N against C2KG-N-blocked")
    tr = triples("c2kg_N", "train")
    te = triples("c2kg_N")
    part, _, _ = partners(tr, te)
    leak = np.array([t in part for t in te])
    _, pat = patterns_of("c2kg_N")
    comp = (pat == "composition") & ~leak
    other = ~leak & ~comp
    say(f"  leaked {leak.sum()}, composition (not leaked) {comp.sum()}, "
        f"other {other.sum()}")
    rows = []
    for m in ORDER:
        a, b = prot(N, m, "s2"), prot(B, m, "s2")
        ks = sorted(set(a) & set(b))
        QA = per_query({k: a[k] for k in ks})
        QB = per_query({k: b[k] for k in ks})
        oa, ob = QA[:, other].mean(1), QB[:, other].mean(1)
        dm, _, p = paired(ob, oa)
        rows.append([NAME[m], f"{QA[:, leak].mean():.3f}",
                     f"{QB[:, leak].mean():.3f}", f"{QA[:, comp].mean():.3f}",
                     f"{QB[:, comp].mean():.3f}", f"{oa.mean():.3f}",
                     f"{ob.mean():.3f}"])
        say(f"  {NAME[m]:9s} leaked {QA[:, leak].mean():.3f} -> "
            f"{QB[:, leak].mean():.3f} | comp {QA[:, comp].mean():.3f} -> "
            f"{QB[:, comp].mean():.3f} | other {oa.mean():.3f} -> "
            f"{ob.mean():.3f} (diff {dm:+.4f}, p={p:.3f})")
    ra, rb = probe_rr("N", "rule"), probe_rr("N_blocked", "rule")
    rows.append(["Rule miner", f"{ra[leak].mean():.3f}", f"{rb[leak].mean():.3f}",
                 f"{ra[comp].mean():.3f}", f"{rb[comp].mean():.3f}",
                 f"{ra[other].mean():.3f}", f"{rb[other].mean():.3f}"])
    say(f"  rule miner leaked {ra[leak].mean():.3f} -> {rb[leak].mean():.3f} | "
        f"comp {ra[comp].mean():.3f} -> {rb[comp].mean():.3f} | other "
        f"{ra[other].mean():.3f} -> {rb[other].mean():.3f}")
    body = "\n".join(" & ".join(r) + r"\\" for r in rows[:7])
    body += "\n\\midrule\n" + " & ".join(rows[7]) + r"\\" + "\n"
    write("tab_same_triples", body)


def section_tuning():
    say("\n== 4. Learning-rate tuning")
    L, N = load("c2kg_L"), load("c2kg_N")
    rows = []
    for m in ORDER[:1] + ORDER[2:]:
        if m == "re_rgcn":
            continue
    order6 = [m for m in ORDER if m != "re_rgcn"]
    for m in order6:
        sl, sn = L["selected"][m], N["selected"][m]
        a, b, c = (v(prot(L, m, w)) for w in ("shared", "s1", "s2"))
        d1, _, p1 = paired(b, a)
        d2, _, p2 = paired(c, b)
        rows.append([NAME[m], f"{fmt_lr(sl['lr1'])} / {fmt_lr(sl['lr2'])}",
                     f"{fmt_lr(sn['lr1'])} / {fmt_lr(sn['lr2'])}",
                     ms(a), ms(b), ms(c)])
        say(f"  {NAME[m]:9s} L rates {sl['lr1']:g}/{sl['lr2']}  N rates "
            f"{sn['lr1']:g}/{sn['lr2']}  shared {ms(a)} (per seed "
            f"{np.round(a, 4).tolist()})  s1 {ms(b)}  s2 {ms(c)}  "
            f"s1-shared {d1:+.4f} p={p1:.2g}  s2-s1 {d2:+.4f} p={p2:.2g}")
    w = L["selected"]["re_rgcn"]
    wn = N["selected"]["re_rgcn"]
    say(f"  WS-RGCN rates L lr1 {w['lr1']} emb {w['emb']} gnn {w['gnn']}; "
        f"N lr1 {wn['lr1']} emb {wn['emb']} gnn {wn['gnn']}")
    for s, R in (("L", L), ("N", N)):
        sh = [v(prot(R, m, "shared")).mean() for m in MODELS]
        tu = [v(prot(R, m, "s2")).mean() for m in MODELS]
        rho = stats.spearmanr(sh, tu).correlation
        say(f"  {s}: shared order " + ", ".join(
            NAME[m] for m in sorted(MODELS, key=lambda m: -sh[MODELS.index(m)]))
            + f"; tuned order " + ", ".join(
            NAME[m] for m in sorted(MODELS, key=lambda m: -tu[MODELS.index(m)]))
            + f"; Spearman {rho:+.2f}")
    be = v(prot(L, "rotate", "s1"), "best")
    bt = v(prot(L, "transe", "s1"), "best")
    say(f"  best epochs stage 1 (L): RotatE {be.tolist()}, TransE {bt.tolist()}")
    write("tab_tuning", "\n".join(" & ".join(r) + r"\\" for r in rows) + "\n")

    # learning-rate curves (stage 1, seed 0, validation MRR)
    say("\n== 4b. Learning-rate curves (C2KG-L, stage 1, seed 0, validation)")
    curves = {}
    for m in MODELS:
        pts = sorted((float(k.split("|")[2][2:]), x["val_mrr"])
                     for k, x in L.items() if "|" in k
                     and k.startswith(f"s1|{m}|") and k.endswith("|s0"))
        curves[m] = pts
        say(f"  {NAME[m]:9s} " + "  ".join(f"{lr:g}:{val:.4f}" for lr, val in pts))
    with open(os.path.join(OUT, "lr_curves.json"), "w", newline="\n") as f:
        json.dump({NAME[m]: c for m, c in curves.items()}, f, indent=1)


def section_budget():
    say("\n== 5. 640 epochs (C2KG-L)")
    E, L = load("ext_L"), load("c2kg_L")
    for m in ("transe", "rotate"):
        s = L["selected"][m]
        sh = v(runs(E, f"s1|{m}|lr{lstr(SHARED)}|"))
        shb = v(runs(E, f"s1|{m}|lr{lstr(SHARED)}|"), "best")
        s1 = v(runs(E, f"s1|{m}|lr{lstr(s['lr1'])}|"))
        s1b = v(runs(E, f"s1|{m}|lr{lstr(s['lr1'])}|"), "best")
        s2 = v(runs(E, f"s2|{m}|lr1{lstr(s['lr1'])}|lr{lstr(s['lr2'])}|"))
        a1 = v(prot(L, m, "s1"))
        a2 = v(prot(L, m, "s2"))
        say(f"  {NAME[m]:7s} shared@640 {ms(sh)} per seed "
            f"{np.round(sh, 3).tolist()} best epochs {shb.tolist()}")
        d, _, p = paired(s1, a1)
        say(f"          s1@640 {ms(s1)} (best epochs {s1b.tolist()}) vs "
            f"s1@160 {ms(a1)}: {d:+.4f} p={p:.2g}")
        d, _, p = paired(s2, s1)
        say(f"          s2 after 640 {ms(s2)}: s2-s1@640 {d:+.4f} p={p:.2g}; "
            f"s2@160 {ms(a2)}")


def section_sensitivity():
    say("\n== 6. Sensitivity (C2KG-L, three seeds, rates selected again)")
    files = [("c2kg_L", "Main (batch 512, $d=64$, $\\gamma=9$)"),
             ("sens_bs4096_L", "Batch 4096"), ("sens_d128_L", "$d=128$"),
             ("sens_m6_L", "$\\gamma=6$"), ("sens_m12_L", "$\\gamma=12$")]
    rows = []
    order6 = [m for m in ORDER if m != "re_rgcn"]
    for f, nm in files:
        R = load(f)
        have = [m for m in order6 if m in R["selected"]]
        cells = []
        for m in order6:
            if m in have:
                cells.append(f"{v(prot(R, m, 's2')).mean():.3f}")
            else:
                cells.append("--")
        if len(have) == 6:
            sh = [v(prot(R, m, "shared")).mean() for m in MODELS]
            tu = [v(prot(R, m, "s2")).mean() for m in MODELS]
            rho = sgn(stats.spearmanr(sh, tu).correlation, 2)
        else:
            rho = "--"
        g = v(prot(R, "rotate", "s2")).mean() - v(prot(R, "rotate", "s1")).mean()
        shr = (v(prot(R, "rotate", "shared")).mean(),
               v(prot(R, "transe", "shared")).mean())
        rows.append([nm] + cells + [f"{shr[1]:.3f} / {shr[0]:.3f}",
                                    sgn(g, 3), rho])
        say(f"  {nm:26s} " + " ".join(f"{NAME[m]} {c}" for m, c in zip(order6, cells))
            + f" | shared TransE/RotatE {shr[1]:.4f}/{shr[0]:.4f} | RotatE s2-s1 "
            f"{g:+.4f} | Spearman {rho}")
        if len(have) == 6:
            say("      shared: " + " ".join(
                f"{NAME[m]} {v(prot(R, m, 'shared')).mean():.4f}" for m in MODELS)
                + " | s1: " + " ".join(
                f"{NAME[m]} {v(prot(R, m, 's1')).mean():.4f}" for m in MODELS)
                + " | rates " + " ".join(
                f"{NAME[m]} {R['selected'][m]['lr1']:g}/{R['selected'][m]['lr2']}"
                for m in MODELS))
    write("tab_sensitivity", "\n".join(" & ".join(r) + r"\\" for r in rows) + "\n")


def section_continuation():
    say("\n== 7. Continuation control (WS-RGCN against RotatE)")
    cases = [("L, graph 42", "c2kg_L"), ("L, graph 43", "c2kg_L_g43"),
             ("L, graph 44", "c2kg_L_g44"), ("N, graph 42", "c2kg_N"),
             ("N, graph 43", "c2kg_N_g43"), ("N, graph 44", "c2kg_N_g44"),
             ("L, graph 42, $\\eta_g = 3 \\times 10^{-4}$", "ws_active_L"),
             ("L, graph 42, batch 4096", "sens_bs4096_L")]
    rows = []
    pool = {"L": [], "N": []}
    gmeans = {"L": [], "N": []}
    for nm, f in cases:
        R = load(f)
        s1 = v(prot(R, "rotate", "s1"))
        s2 = v(prot(R, "rotate", "s2"))
        ws = v(prot(R, "re_rgcn", "ws"))
        gc = v(prot(R, "re_rgcn", "ws"), "gnn")
        d2 = ws - s2
        d1 = ws - s1
        p_t, ci = tost(d2)
        _, _, p1 = paired(ws, s1)
        enc = (f"{gc.min():.2f}--{gc.max():.2f}" if gc.max() >= 0.01
               else "$<$0.01")
        rows.append([nm, str(len(ws)), f"{s1.mean():.4f}", f"{s2.mean():.4f}",
                     f"{ws.mean():.4f}", sgn(d1.mean()), sgn(d2.mean()),
                     pfmt(p_t), enc])
        say(f"  {nm:32s} n={len(ws)} s1 {s1.mean():.4f} s2 {s2.mean():.4f} "
            f"WS {ws.mean():.4f} | WS-s1 {d1.mean():+.4f} (p={p1:.2g}) | WS-s2 "
            f"{d2.mean():+.4f} sd {d2.std(ddof=1):.4f} TOST p={p_t:.2g} 90% CI "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | encoder change "
            f"{gc.min():.2f}-{gc.max():.2f}")
        if f.startswith("c2kg_"):
            sp = nm[0]
            pool[sp] += d2.tolist()
            gmeans[sp].append(d2.mean())
    for sp in ("L", "N"):
        d = np.array(pool[sp])
        lo, hi = ci95(d)
        p_t, _ = tost(d)
        pg, cig = tost(gmeans[sp])
        say(f"  pooled {sp}: n={len(d)} mean {d.mean():+.4f} 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] TOST p={p_t:.2g}; graph means "
            f"{np.round(gmeans[sp], 4).tolist()} graph-level TOST p={pg:.2g}")
    write("tab_continuation", "\n".join(" & ".join(r) + r"\\" for r in rows) + "\n")


def section_instances():
    say("\n== 8. Graph instances")
    for g, fl, fn, pl, pn in (("42", "c2kg_L", "c2kg_N", "L", "N"),
                              ("43", "c2kg_L_g43", "c2kg_N_g43", "L_g43", "N_g43"),
                              ("44", "c2kg_L_g44", "c2kg_N_g44", "L_g44", "N_g44")):
        L, N = load(fl), load(fn)
        tl = {m: v(prot(L, m, "s2")).mean() for m in M7}
        tn = {m: v(prot(N, m, "s2")).mean() for m in M7}
        shl = {m: v(prot(L, m, "shared")).mean() for m in MODELS}
        rho = stats.spearmanr([shl[m] for m in MODELS],
                              [tl[m] for m in MODELS]).correlation
        ratio = {m: tn[m] / tl[m] for m in M7}
        say(f"  graph {g}: seeds {len(prot(L, 'rotate', 's2'))}; tuned L "
            + " ".join(f"{NAME[m]} {tl[m]:.3f}" for m in ORDER))
        say(f"           tuned N " + " ".join(f"{NAME[m]} {tn[m]:.3f}" for m in ORDER))
        say(f"           N/L " + " ".join(f"{NAME[m]} {ratio[m]:.1f}" for m in ORDER))
        say(f"           shared L " + " ".join(f"{NAME[m]} {shl[m]:.4f}" for m in MODELS)
            + f"; Spearman shared vs tuned {rho:+.2f}")
        say(f"           order L: " + ", ".join(NAME[m] for m in sorted(M7, key=lambda m: -tl[m]))
            + " | order N: " + ", ".join(NAME[m] for m in sorted(M7, key=lambda m: -tn[m])))
        say(f"           rule miner L {probe(pl)['rule']['overall']['MRR']:.4f} "
            f"N {probe(pn)['rule']['overall']['MRR']:.4f}")


def section_rule_vs_models():
    say("\n== 9. Rule miner against the best trained model (paired over test "
        "triples, model RR averaged over seeds)")
    for nm, f, pr in (("C2KG-L 42", "c2kg_L", "L"), ("C2KG-L 43", "c2kg_L_g43", "L_g43"),
                      ("C2KG-L 44", "c2kg_L_g44", "L_g44"), ("C2KG-N 42", "c2kg_N", "N"),
                      ("C2KG-N 43", "c2kg_N_g43", "N_g43"), ("C2KG-N 44", "c2kg_N_g44", "N_g44"),
                      ("C2KG-N-blocked", "c2kg_Nb", "N_blocked")):
        R = load(f)
        rq = probe_rr(pr, "rule")
        best = max(M7, key=lambda m: v(prot(R, m, "s2")).mean())
        Q = per_query(prot(R, best, "s2")).mean(0)
        d = rq - Q
        lo, hi = boot(d)
        say(f"  {nm:15s} rule {rq.mean():.4f}  best {NAME[best]} {Q.mean():.4f}  "
            f"rule-best {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")


def section_wn18rr():
    say("\n== 10. WN18RR")
    te = triples("wn18rr_O")
    tr = triples("wn18rr_O", "train")
    part, _, _ = partners(tr, te)
    leak = np.array([t in part for t in te])
    btr = set(triples("wn18rr_O_blocked", "train"))
    still = np.array([t in part and any(p in btr for p in part[t]) for t in te])
    seen = {e for h, _, t in tr for e in (h, t)}
    unseen = np.array([(h not in seen) or (t not in seen) for h, _, t in te])
    _, pat = patterns_of("wn18rr_O")
    say(f"  test {len(te)}; mirror in training {leak.sum()} "
        f"({100 * leak.mean():.1f}%); kept after blocking {still.sum()}; "
        f"unseen entity {unseen.sum()}; symmetric relations {int((pat == 'symmetric').sum())}")
    rows = []
    for m in ("rotate", "complex", "distmult"):
        for split, f, folder in (("Original", "wn_O", "wn18rr_O"),
                                 ("Original, mirrors removed", "wn_Ob", "wn18rr_O_blocked"),
                                 ("Leakage-aware", "wn_L", "wn18rr_L")):
            R = load(f)
            d = prot(R, m, "s2")
            _, pt = patterns_of(folder)
            Q = per_query(d)
            sym = Q[:, pt == "symmetric"].mean()
            oth = Q[:, pt != "symmetric"].mean()
            rows.append([NAME[m] if split == "Original" else "", split,
                         ms(v(d, "ht")), f"{v(d, 'ht_h1').mean():.3f}",
                         f"{v(d, 'ht_h10').mean():.3f}", f"{v(d).mean():.3f}",
                         f"{sym:.3f}", f"{oth:.3f}"])
            say(f"  {NAME[m]:8s} {split:26s} h+t {ms(v(d, 'ht'))} H@1 "
                f"{v(d, 'ht_h1').mean():.3f} H@10 {v(d, 'ht_h10').mean():.3f} | "
                f"tail {ms(v(d))} sym {sym:.3f} other {oth:.3f} | seeds {len(d)}")
        O, B = prot(load("wn_O"), m, "s2"), prot(load("wn_Ob"), m, "s2")
        ks = sorted(set(O) & set(B))
        QO = per_query({k: O[k] for k in ks})
        QB = per_query({k: B[k] for k in ks})
        blk = leak & ~still
        QOh = per_query({k: O[k] for k in ks}, "head")
        say(f"           same triples: blocked {blk.sum()} {QO[:, blk].mean():.3f} -> "
            f"{QB[:, blk].mean():.3f}; not leaked {(~leak).sum()} "
            f"{QO[:, ~leak].mean():.3f} -> {QB[:, ~leak].mean():.3f}; "
            f"O without unseen-entity triples: tail {QO[:, ~unseen].mean():.4f}, "
            f"head+tail {np.concatenate([QO[:, ~unseen], QOh[:, ~unseen]], 1).mean():.4f}")
    for pr, nm in (("wn_O", "Original"), ("wn_O_blocked", "mirrors removed"),
                   ("wn_L", "leakage-aware")):
        P = probe(pr)
        say(f"  rule miner {nm}: tail {P['rule']['overall']['MRR']:.4f} "
            f"({P['rule']['selected']['aggregation']}); frequency "
            f"{P['freq']['overall']['MRR']:.4f}")
    body = ""
    for i, r in enumerate(rows):
        if i and i % 3 == 0:
            body += "\\midrule\n"
        body += " & ".join(r) + r"\\" + "\n"
    write("tab_wn18rr", body)
    # per-relation sample sizes
    say("  per relation: n(O) n(L) | RotatE tail MRR O / Ob / L")
    relO, _ = patterns_of("wn18rr_O")
    relL, _ = patterns_of("wn18rr_L")
    QO = per_query(prot(load("wn_O"), "rotate", "s2"))
    QB = per_query(prot(load("wn_Ob"), "rotate", "s2"))
    QL = per_query(prot(load("wn_L"), "rotate", "s2"))
    rows = []
    for r in sorted(set(relO) | set(relL), key=lambda r: -(relO == r).sum()):
        a, b = relO == r, relL == r
        cells = ["\\texttt{" + r.replace("_", "\\_") + "}",
                 str(int(a.sum())), str(int(b.sum())),
                 f"{QO[:, a].mean():.3f}" if a.any() else "--",
                 f"{QB[:, a].mean():.3f}" if a.any() else "--",
                 f"{QL[:, b].mean():.3f}" if b.any() else "--"]
        rows.append(cells)
        say(f"    {r:30s} {a.sum():5d} {b.sum():5d} | " + " / ".join(cells[3:]))
    write("tab_wn_relations", "\n".join(" & ".join(x) + r"\\" for x in rows) + "\n")


def section_testsets():
    say("\n== 11. Test sets by pattern")
    for folder in ("c2kg_N", "c2kg_L"):
        rel, pat = patterns_of(folder)
        same = (pat == "antisymmetric") & np.isin(rel, list(SAME_TYPE_ANTI))
        typed = (pat == "antisymmetric") & ~same
        c = Counter(pat)
        say(f"  {folder}: " + " ".join(f"{p} {c[p]}" for p in PATS)
            + f" | antisymmetric same {same.sum()} typed {typed.sum()}"
            + f" | commands in test {(rel == 'commands').sum()}")


def main():
    section_checks()
    section_testsets()
    section_reference()
    section_patterns()
    section_leakage()
    section_tuning()
    section_budget()
    section_sensitivity()
    section_continuation()
    section_instances()
    section_rule_vs_models()
    section_wn18rr()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "numbers.txt"), "w", newline="\n") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
