"""
c2kg_runner.py
--------------
One runner for every C2KG link-prediction experiment, on CPU or GPU.  All
trained-model results come from this file, so that every model is trained
and tuned by the same code path on the same device.

Protocol (identical for every model)
  naive   stage 1 at lr 1e-2 for every model: a single shared rate.
          Used only for the tuning-confound comparison.
  s1      stage 1: train from scratch at the model's selected rate lr1.
  s2      stage 2: continue from the final stage-1 model at rate lr2 with
          a fresh optimiser.  This is the continuation control.
  ws      warm-started encoder (re_rgcn): residual R-GCN on top of the
          stage-1 RotatE.  Embeddings and relation phases at RotatE's lr2,
          GNN weights at their own rate.  Compared against RotatE s2,
          from which it differs only by the relational blocks.

Rate selection (seed 0, validation MRR, never test)
  Rates come from a log ladder 1e-4, 3e-4, 1e-3, ..., 1e1, 3e1.  Each
  search starts from a model-specific range; whenever the best rate is at
  the edge of the rates tried, the next rung outward is tried
  automatically, so no selected rate sits on a boundary.  For stage 2 the
  option "no stage 2" competes too: if no stage-2 rate beats the stage-1
  validation MRR, lr2 is recorded as none.

Fixed settings: d=64, margin 9, at most 160 epochs, batch 512, evaluation
every 15 epochs, patience 6, KvsAll BCE with label smoothing 0.1
(train_eval.train_model).  --dim, --margin, --bs and --epochs change them
for sensitivity runs; they are stored in the results file and checked by
the fingerprint, so different settings never share one file.

Metrics: filtered ranking; MRR and Hits@k are exact expectations under
uniformly random tie-breaking (see train_eval.py), MR the expected rank.

Subcommands (run from the code directory)
  smoke   2 epochs per model: checks that the device works end to end
  sweep   seed 0: stage-1, stage-2 and ws rate selection
  final   all seeds at the selected rates (--rates-from: reuse the rates
          selected on another dataset, for the extra graph instances)
  report  tables from a results file

Example (C2KG-L = leakage-aware split, C2KG-N = inverse-pair split)
  python c2kg_runner.py smoke  --data ../data/c2kg_L --device cuda
  python c2kg_runner.py sweep  --data ../data/c2kg_L --out ../runs/c2kg_L.json --device cuda
  python c2kg_runner.py final  --data ../data/c2kg_L --out ../runs/c2kg_L.json --device cuda
  python c2kg_runner.py report --out ../runs/c2kg_L.json

Every finished run is written to the results file at once; re-running a
command skips what is already there, so an interrupted run resumes.

Safeguards
  * selftest: before smoke, sweep and final, the evaluator is checked on
    scorers whose correct MRR is known exactly (oracle = 1.0; constant and
    all-NaN scores = the random-ranking value; anti-oracle = 1/(#non-true
    candidates + 1)) and on hand-computed tie arithmetic.  Any failure
    stops.
  * fingerprint: the first run stores hashes of the data files and of the
    code, plus device and library versions, in the results file.  A later
    run with different data, different code or a different device stops
    instead of mixing results.
  * tripwire: a run whose validation or test MRR exceeds 0.9 stops the
    command; no model can legitimately reach that on C2KG.
  * degenerate runs (training diverged, or on the validation set >5 % of
    queries with a tied gold score, or any NaN score) are recorded but can
    never be selected as a learning rate.  The test set is used for no
    decision.
  * the results file is written atomically.

Requires train_eval.py, models.py, ontology.py and warmstart_gnn.py in the
same directory.  No file on disk is modified.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import time

import numpy as np
import torch

import models
import train_eval
from train_eval import load_dicts, load_triples, build_adj
from models import build_model as _orig_build_model
from ontology import build_type_mask
from warmstart_gnn import build_warmstart

# ---- fixed protocol ----------------------------------------------------------
DIM, EPOCHS, BS, EVAL_EVERY, PATIENCE = 64, 160, 512, 15, 6
MODELS = ["transe", "rotate", "distmult", "complex", "rgcn", "compgcn"]
WS = "re_rgcn"
METRIC = "expected-rr-v2"          # exact expectations under random ties
NAIVE_LR = 1e-2
PATS = ["symmetric", "antisymmetric", "inverse", "composition", "N-N",
        "general"]
LADDER = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1,
          1e0, 3e0, 1e1, 3e1]
START_S1 = {"transe": (1e-2, 1e1), "rotate": (1e-2, 1e1),
            "distmult": (1e-3, 3e-1), "complex": (1e-3, 3e-1),
            "rgcn": (1e-3, 1e-1), "compgcn": (1e-3, 1e-1)}
S2_BELOW, S2_ABOVE = 4, 2          # stage-2 start: rungs around lr1
START_WS = (3e-4, 3e-2)


def rung(x):
    for i, v in enumerate(LADDER):
        if math.isclose(x, v, rel_tol=1e-6):
            return i
    raise ValueError(f"{x} is not on the ladder")


def lr_str(x):
    if x is None:
        return "none"
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def ladder_range(lo, hi):
    return LADDER[rung(lo):rung(hi) + 1]


# ---- in-process hooks into train_eval (nothing on disk is changed) ------------
_STATE = {"init": None, "emb_lr": None}


def _build_model(name, n_ent, n_rel, dim=200, num_bases=10, dropout=0.2):
    if name == WS:
        m = build_warmstart(WS, n_ent, n_rel, dim, num_bases=0,
                            dropout=dropout)
    else:
        m = _orig_build_model(name, n_ent, n_rel, dim,
                              num_bases=num_bases, dropout=dropout)
    if _STATE["init"] is not None:          # stage-2 state or warm start
        _STATE["init"](m)
    return m


def _param_groups(model, lr):
    blocks = getattr(model, "blocks", None)
    if _STATE["emb_lr"] is None or not blocks:
        return model.parameters()            # every model except ws
    gnn = list(blocks.parameters())
    ids = {id(p) for p in gnn}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": rest, "lr": _STATE["emb_lr"]},
            {"params": gnn, "lr": lr}]


def _install_hooks():
    src_all = inspect.getsource(train_eval)
    if "tail-only patch" not in src_all:
        raise SystemExit("train_eval.py lacks the tail-only evaluation; "
                         "use the train_eval.py of this repository")
    if "tie/NaN fix" not in src_all or "[guard]" not in src_all:
        raise SystemExit("train_eval.py is not the version of this "
                         "repository")
    anchor = "opt = torch.optim.Adam(model.parameters(), lr=lr)"
    src = inspect.getsource(train_eval.train_model)
    if src.count(anchor) != 1:
        raise SystemExit("train_eval.train_model: optimiser line not found "
                         "exactly once; this train_eval.py is not supported")
    train_eval.build_model = _build_model
    train_eval._c2kg_param_groups = _param_groups
    exec(src.replace(anchor, "opt = torch.optim.Adam("
                             "_c2kg_param_groups(model, lr), lr=lr)"),
         train_eval.__dict__)


# ---- context -----------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.args = args
        self.data = args.data
        self.device = args.device
        self.epochs = args.epochs
        e2i, r2i = load_dicts(args.data)
        self.n_ent, self.n_rel = len(e2i), len(r2i)
        self.masks = build_type_mask({v: k for k, v in e2i.items()}, r2i)
        self.out = args.out
        self.R = (json.load(open(args.out))
                  if args.out and os.path.exists(args.out) else {})
        if args.out:
            d = os.path.dirname(os.path.abspath(args.out))
            self.ckdir = os.path.join(d, "ckpt")
            os.makedirs(self.ckdir, exist_ok=True)
            self.stem = os.path.splitext(os.path.basename(args.out))[0]
        self._adj = None
        self.t0 = time.time()

    @property
    def adj(self):
        if self._adj is None:
            tr = load_triples(os.path.join(self.data, "train.txt"),
                              *load_dicts(self.data))
            self._adj = build_adj(tr, self.n_ent, self.n_rel, self.device)
        return self._adj

    def save(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.R, f, indent=2)
        os.replace(tmp, self.out)          # atomic: never a half-written file

    def state_path(self, model, lr1, seed):
        return os.path.join(self.ckdir,
                            f"{self.stem}_s1_{model}_lr{lr_str(lr1)}"
                            f"_s{seed}.pt")

    def elapsed(self):
        return f"{time.time() - self.t0:.0f}s"


def summarise(res):
    t = res["test"]
    tail = t.get("tail_only", {})
    pat = res["pattern"] or {}
    return {"val_mrr": res["val_mrr"],
            "test_MRR": t["MRR"],
            "tail_MRR": tail.get("MRR"),
            "tail_H@1": tail.get("H@1"), "tail_H@3": tail.get("H@3"),
            "tail_H@10": tail.get("H@10"),
            "pattern_tail": {p: v.get("MRR_tail") for p, v in pat.items()},
            "params": res["params"], "time": res["train_time_s"],
            "best_epoch": res.get("best_epoch"),
            "last_epoch": res.get("last_epoch"),
            "diverged_epoch": res.get("diverged_epoch"),
            "val_tied_frac": res.get("val_tied_frac"),
            "val_nan_frac": res.get("val_nan_frac"),
            "test": t, "pattern": pat}


IMPLAUSIBLE = 0.9          # no model reaches this MRR on C2KG legitimately


def degenerate(rec):
    """Runs that must never be selected: diverged, collapsed or NaN.
    Judged on the validation set, so that the test set enters no decision."""
    return (rec.get("diverged_epoch") is not None
            or (rec.get("val_tied_frac") or 0) > 0.05
            or (rec.get("val_nan_frac") or 0) > 0)


def sel_val(rec):
    return -1.0 if degenerate(rec) else rec["val_mrr"]


def train(ctx, name, lr, seed, init=None, emb_lr=None, epochs=None):
    _STATE["init"], _STATE["emb_lr"] = init, emb_lr
    try:
        res = train_eval.train_model(
            name, ctx.data, dim=DIM, epochs=epochs or ctx.epochs, lr=lr,
            seed=seed, device=ctx.device, eval_every=EVAL_EVERY,
            patience=PATIENCE, bs=BS, type_masks=ctx.masks)
    finally:
        _STATE["init"], _STATE["emb_lr"] = None, None
    return res


def load_state(path):
    return torch.load(path, map_location="cpu")


def show(key, rec, ctx, extra=""):
    pt = rec["pattern_tail"]
    comp = pt.get("composition")
    tied, nan = rec["test"].get("tied_frac", 0), rec["test"].get("nan_frac", 0)
    warn = (f"  !! tied {tied:.1%}" if tied > 0.01 else "") + \
           (f"  !! NaN {nan:.1%}" if nan > 0 else "")
    if rec.get("diverged_epoch") is not None:
        warn += f"  !! diverged at epoch {rec['diverged_epoch']}"
    print(f"    val {rec['val_mrr']:.4f}  tail {rec['tail_MRR']:.4f}"
          + (f"  comp {comp:.3f}" if comp is not None else "")
          + f"{extra}{warn}   ({ctx.elapsed()})", flush=True)
    if max(rec["val_mrr"], rec["tail_MRR"]) > IMPLAUSIBLE:
        raise SystemExit(
            f"\nSTOPPED: {key} reports MRR above {IMPLAUSIBLE}, which is not "
            f"plausible on C2KG.\nThe record is kept in the results file. "
            f"Send this output before running anything else.")


# ---- the three kinds of run --------------------------------------------------
def run_s1(ctx, model, lr1, seed, keep_state=True):
    key = f"s1|{model}|lr{lr_str(lr1)}|s{seed}"
    path = ctx.state_path(model, lr1, seed)
    if key in ctx.R and (not keep_state or os.path.exists(path)):
        return ctx.R[key]
    print(f">>> {key}", flush=True)
    res = train(ctx, model, lr1, seed)
    m = res.pop("trained_model")
    if keep_state:
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()},
                   path)
    ctx.R[key] = summarise(res)
    ctx.save()
    show(key, ctx.R[key], ctx)
    return ctx.R[key]


def run_s2(ctx, model, lr1, lr2, seed):
    key = f"s2|{model}|lr1{lr_str(lr1)}|lr{lr_str(lr2)}|s{seed}"
    if key in ctx.R:
        return ctx.R[key]
    run_s1(ctx, model, lr1, seed)
    state = load_state(ctx.state_path(model, lr1, seed))
    print(f">>> {key}", flush=True)
    res = train(ctx, model, lr2, seed,
                init=lambda m: m.load_state_dict(state))
    res.pop("trained_model")
    ctx.R[key] = summarise(res)
    ctx.save()
    show(key, ctx.R[key], ctx)
    return ctx.R[key]


@torch.no_grad()
def _identity_err(ws, base, adj, n_ent, n_rel, device):
    g = torch.Generator().manual_seed(123)
    h = torch.randint(0, n_ent, (256,), generator=g).to(device)
    r = torch.randint(0, n_rel, (256,), generator=g).to(device)
    ws.eval(); base.eval()
    E, Eb = ws.encode(adj), base.encode()
    return float(max(
        (ws.score_all_t(E, h, r) - base.score_all_t(Eb, h, r)).abs().max(),
        (ws.score_all_h(E, r, h) - base.score_all_h(Eb, r, h)).abs().max()))


def run_ws(ctx, lr1, emb_lr, gnn_lr, seed):
    key = (f"ws|{WS}|lr1{lr_str(lr1)}|emb{lr_str(emb_lr)}"
           f"|lr{lr_str(gnn_lr)}|s{seed}")
    if key in ctx.R:
        return ctx.R[key]
    run_s1(ctx, "rotate", lr1, seed)
    pre = _orig_build_model("rotate", ctx.n_ent, ctx.n_rel, DIM)
    pre.load_state_dict(load_state(ctx.state_path("rotate", lr1, seed)))
    pre.eval()
    probe = build_warmstart(WS, ctx.n_ent, ctx.n_rel, DIM)
    probe.init_from(pre)
    ierr = _identity_err(probe.to(ctx.device), pre.to(ctx.device), ctx.adj,
                         ctx.n_ent, ctx.n_rel, ctx.device)
    pre = pre.cpu()
    del probe
    print(f">>> {key}", flush=True)
    res = train(ctx, WS, gnn_lr, seed, init=lambda m: m.init_from(pre),
                emb_lr=emb_lr)
    m = res.pop("trained_model")
    rec = summarise(res)
    rec["identity_err"] = ierr
    rec["gnn_change"] = m.gnn_change(ctx.adj)
    ctx.R[key] = rec
    ctx.save()
    show(key, rec, ctx, f"  gnn {rec['gnn_change']:.3f}  id-err {ierr:.0e}")
    return rec


# ---- safeguards: evaluator self-test, data checks, fingerprint ---------------
def _sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def code_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    return _sha([os.path.join(here, f) for f in
                 ("c2kg_runner.py", "train_eval.py", "models.py",
                  "warmstart_gnn.py", "ontology.py")])


def data_sha(data):
    files = ["entities.dict", "relations.dict", "train.txt", "valid.txt",
             "test.txt", "relation_patterns.json"]
    if os.path.exists(os.path.join(data, train_eval.FILTER_EXTRA)):
        files.append(train_eval.FILTER_EXTRA)       # leak-blocked variant
    return _sha([os.path.join(data, f) for f in files])


def selftest(data, device):
    """Check the evaluator on models whose correct answer is known."""
    e2i, r2i = load_dicts(data)
    n = len(e2i)
    tr = load_triples(os.path.join(data, "train.txt"), e2i, r2i)
    va = load_triples(os.path.join(data, "valid.txt"), e2i, r2i)
    te = load_triples(os.path.join(data, "test.txt"), e2i, r2i)
    fails = []

    # data: no held-out triple may also be a training triple
    trs = {tuple(x) for x in tr.tolist()}
    if any(tuple(x) in trs for x in te.tolist()) or \
       any(tuple(x) in trs for x in va.tolist()):
        fails.append("a valid/test triple also occurs in train")

    xp = os.path.join(data, train_eval.FILTER_EXTRA)
    ex = (load_triples(xp, e2i, r2i) if os.path.exists(xp)
          else np.zeros((0, 3), np.int64))
    tf, hf = train_eval.all_true(tr, va, te, ex)

    class Fixed(torch.nn.Module):
        encoder_kind = "none"

        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        def encode(self):
            return torch.zeros(n, 1, device=device)

        def _rows(self, keys, table):
            out = torch.zeros(len(keys), n, device=device)
            for i, k in enumerate(keys):
                if self.kind == "oracle":       # every known true scores 1
                    out[i, list(table[k])] = 1.0
                elif self.kind == "anti":       # every known true scores -1
                    out[i, list(table[k])] = -1.0
            if self.kind == "nan":
                out[:] = float("nan")
            return out

        def score_all_t(self, E, h, r):
            return self._rows([(int(a), int(b)) for a, b in zip(h, r)], tf)

        def score_all_h(self, E, r, t):
            return self._rows([(int(a), int(b)) for a, b in zip(r, t)], hf)

    # exact expected MRR of each scorer (tail and head queries)
    Hn = [0.0]
    for k in range(1, n + 1):
        Hn.append(Hn[-1] + 1.0 / k)
    sub = te[:200]
    sizes = ([len(tf[(int(h), int(r))]) for h, r, t in sub]
             + [len(hf[(int(r), int(t))]) for h, r, t in sub])
    expect = {"oracle": 1.0,
              # all candidates tie except the filtered other true ones
              "const": float(np.mean([Hn[n - k + 1] / (n - k + 1)
                                      for k in sizes])),
              # every non-true candidate scores above the gold
              "anti": float(np.mean([1.0 / (n - k + 1) for k in sizes])),
              # all scores -inf: the gold ties with every entity
              "nan": Hn[n] / n}
    for kind, want in expect.items():
        res, _ = train_eval.evaluate(Fixed(kind), sub, None, tf, hf, n,
                                     device)
        if abs(res["MRR"] - want) > 1e-9:
            fails.append(f"{kind}: MRR {res['MRR']:.9f}, expected {want:.9f}")
    # tie arithmetic by hand: gold 3, one candidate above, two equal ->
    # ranks 2..4 equally likely
    H5 = train_eval._harmonic(5, "cpu")
    g, m = train_eval._tie_counts(torch.tensor([[5., 3., 3., 3., 1.]]),
                                  torch.tensor([3.]))
    ex_ = train_eval._expected(g, m, H5)
    want = {"rr": (1 / 2 + 1 / 3 + 1 / 4) / 3, "rank": 3.0,
            "h1": 0.0, "h3": 2 / 3, "h10": 1.0}
    for k_, v_ in want.items():
        if abs(float(ex_[k_][0]) - v_) > 1e-12:
            fails.append(f"tie arithmetic {k_}: {float(ex_[k_][0])}, "
                         f"expected {v_}")
    if fails:
        raise SystemExit("SELFTEST FAILED -- nothing was run:\n  "
                         + "\n  ".join(fails))
    print("selftest ok: oracle 1.0, constant/NaN = random-ranking value, "
          "anti-oracle exact, tie arithmetic exact, no train/test overlap")


def check_fingerprint(ctx, allow_change):
    meta = {"data_sha": data_sha(ctx.data), "code_sha": code_sha(),
            "device": ("cuda:" + torch.cuda.get_device_name(0))
            if ctx.device.startswith("cuda") else "cpu",
            "torch": torch.__version__, "n_ent": ctx.n_ent,
            "epochs": ctx.epochs, "bs": BS, "dim": DIM,
            "margin": models.MARGIN, "metric": METRIC,
            "data": os.path.abspath(ctx.data)}
    old = ctx.R.get("_meta")
    if old is None:
        ctx.R["_meta"] = meta
        ctx.save()
        return
    diff = [k for k in ("data_sha", "code_sha", "device", "torch", "epochs",
                        "bs", "dim", "margin", "metric")
            if old.get(k) != meta[k]]
    if diff and not allow_change:
        raise SystemExit(
            f"STOPPED: {ctx.out} was produced with a different "
            f"{', '.join(diff)}.\nMixing would make the results "
            f"inconsistent. Use a new --out file (or ask before overriding).")
    if diff:
        ctx.R.setdefault("_meta_changes", []).append(meta)
        ctx.save()


# ---- rate search with automatic edge extension --------------------------------
def search(start_rates, val_of, label):
    tried = {}
    for lr in start_rates:
        tried[lr] = val_of(lr)
    while True:
        best = max(tried, key=tried.get)
        lo, hi = min(tried), max(tried)
        nxt = None
        if math.isclose(best, hi) and rung(hi) + 1 < len(LADDER):
            nxt = LADDER[rung(hi) + 1]
        elif math.isclose(best, lo) and rung(lo) > 0:
            nxt = LADDER[rung(lo) - 1]
        if nxt is None or nxt in tried:
            break
        print(f"    [{label}] best {lr_str(best)} is at the edge; "
              f"extending to {lr_str(nxt)}", flush=True)
        tried[nxt] = val_of(nxt)
    best = max(tried, key=tried.get)
    edge = (math.isclose(best, LADDER[0]) or math.isclose(best, LADDER[-1]))
    return best, tried, edge


def cmd_sweep(ctx):
    seed = 0
    sel = ctx.R.setdefault("selected", {})
    for model in ctx.args.models:
        # stage 1 (the naive rate 1e-2 is always among the rates tried)
        lo, hi = START_S1[model]
        lr1, tried1, edge1 = search(
            ladder_range(lo, hi),
            lambda lr: sel_val(run_s1(ctx, model, lr, seed)),
            f"{model} s1")
        # stage 2: rungs around lr1, plus the option of no stage 2
        i = rung(lr1)
        start2 = LADDER[max(0, i - S2_BELOW):min(len(LADDER), i + S2_ABOVE + 1)]
        lr2, tried2, edge2 = search(
            start2,
            lambda lr: sel_val(run_s2(ctx, model, lr1, lr, seed)),
            f"{model} s2")
        if tried2[lr2] <= tried1[lr1]:
            lr2 = None
        sel[model] = {"lr1": lr1, "lr2": lr2, "val1": tried1[lr1],
                      "val2": tried2.get(lr2) if lr2 else None,
                      "edge": bool(edge1 or edge2)}
        ctx.save()
        print(f"  == {model}: lr1 {lr_str(lr1)}  lr2 {lr_str(lr2)}\n",
              flush=True)

    if WS in ctx.args.models_ws and "rotate" in sel:
        lr1 = sel["rotate"]["lr1"]
        emb = sel["rotate"]["lr2"] or lr1
        g, triedg, edgeg = search(
            ladder_range(*START_WS),
            lambda lr: sel_val(run_ws(ctx, lr1, emb, lr, seed)),
            f"{WS} gnn")
        sel[WS] = {"lr1": lr1, "emb": emb, "gnn": g, "val": triedg[g],
                   "edge": bool(edgeg)}
        ctx.save()
        print(f"  == {WS}: emb {lr_str(emb)}  gnn {lr_str(g)}", flush=True)
    print(f"\nSWEEP DONE ({ctx.elapsed()})")


def cmd_final(ctx):
    src = ctx.args.rates_from
    if src:                          # e.g. other graph instances (G5)
        other = json.load(open(src))
        if "selected" not in other:
            raise SystemExit(f"{src} has no selected rates; run sweep there")
        if ctx.R.get("selected_from") not in (None, src):
            raise SystemExit(f"{ctx.out} already uses rates from "
                             f"{ctx.R['selected_from']}")
        ctx.R["selected"] = other["selected"]
        ctx.R["selected_from"] = src
        ctx.save()
        print(f"rates taken from {src} (no sweep on this dataset)")
    sel = ctx.R.get("selected", {})
    missing = [m for m in ctx.args.models if m not in sel]
    if missing:
        raise SystemExit(f"run sweep first; no selected rates for {missing}")
    for seed in ctx.args.seeds:
        for model in ctx.args.models:
            s = sel[model]
            run_s1(ctx, model, NAIVE_LR, seed,
                   keep_state=math.isclose(s["lr1"], NAIVE_LR))
            run_s1(ctx, model, s["lr1"], seed)
            if s["lr2"] is not None:
                run_s2(ctx, model, s["lr1"], s["lr2"], seed)
        if WS in ctx.args.models_ws and WS in sel:
            w = sel[WS]
            run_ws(ctx, w["lr1"], w["emb"], w["gnn"], seed)
        print(f"  seed {seed} done ({ctx.elapsed()})", flush=True)
    print(f"\nFINAL DONE ({ctx.elapsed()})")


def cmd_smoke(ctx):
    print("2-epoch run of every model; numbers are meaningless here.")
    for model in MODELS:
        t = time.time()
        res = train(ctx, model, 1e-2, 0, epochs=2)
        print(f"  {model:9s} ok  tail {res['test']['tail_only']['MRR']:.4f}"
              f"  {time.time() - t:5.1f}s", flush=True)
        if model == "rotate":
            pre = res["trained_model"].cpu().eval()
    t = time.time()
    res = train(ctx, WS, 1e-3, 0, epochs=2, emb_lr=1e-2,
                init=lambda m: m.init_from(pre))
    print(f"  {WS:9s} ok  tail {res['test']['tail_only']['MRR']:.4f}"
          f"  {time.time() - t:5.1f}s  "
          f"gnn {res['trained_model'].gnn_change(ctx.adj):.3f}", flush=True)
    print("SMOKE OK")


# ---- report ------------------------------------------------------------------
def _runs(R, prefix):
    return {k: v for k, v in R.items() if k.startswith(prefix + "|")}


def cmd_report(ctx):
    R = ctx.R
    sel = R.get("selected", {})
    print("== rate selection (seed 0, validation MRR) ==")
    for model in MODELS:
        if model not in sel:
            continue
        s = sel[model]
        s1 = sorted((float(k.split("|")[2][2:]), v["val_mrr"])
                    for k, v in _runs(R, "s1").items()
                    if k.split("|")[1] == model and k.endswith("|s0"))
        s2 = sorted((float(k.split("|")[3][2:]), v["val_mrr"])
                    for k, v in _runs(R, "s2").items()
                    if k.split("|")[1] == model and k.endswith("|s0")
                    and k.split("|")[2] == f"lr1{lr_str(s['lr1'])}")
        f = lambda xs, b: "  ".join(
            f"{lr_str(lr)}:{v:.4f}{'*' if b is not None and math.isclose(lr, b) else ''}"
            for lr, v in xs)
        print(f"{model:9s} s1  {f(s1, s['lr1'])}")
        print(f"{'':9s} s2  {f(s2, s['lr2'])}"
              + ("   (no stage 2 selected)" if s["lr2"] is None else "")
              + ("   !! edge of ladder" if s.get("edge") else ""))
    if WS in sel:
        w = sel[WS]
        g = sorted((float(k.split("|")[4][2:]), v["val_mrr"])
                   for k, v in _runs(R, "ws").items() if k.endswith("|s0"))
        print(f"{WS:9s} gnn " + "  ".join(
            f"{lr_str(lr)}:{v:.4f}{'*' if math.isclose(lr, w['gnn']) else ''}"
            for lr, v in g) + f"   (emb {lr_str(w['emb'])})")

    def coll(keyfn):
        xs = [v for k, v in R.items() if keyfn(k)]
        return xs

    def ms(xs, f):
        a = np.array([f(x) for x in xs], dtype=float)
        return (f"{a.mean():.4f}+/-{a.std(ddof=1) if len(a) > 1 else 0:.4f}"
                f" (n={len(a)})") if len(a) else "-"

    print("\n== final: tail-only test MRR, mean +/- std over seeds ==")
    print(f"{'model':9s} {'naive 1e-2':>22s} {'tuned s1':>22s} "
          f"{'two-stage s2':>22s}")
    for model in MODELS:
        if model not in sel:
            continue
        s = sel[model]
        nv = coll(lambda k: k.startswith(f"s1|{model}|lr{lr_str(NAIVE_LR)}|"))
        t1 = coll(lambda k: k.startswith(f"s1|{model}|lr{lr_str(s['lr1'])}|"))
        t2 = (coll(lambda k: k.startswith(
            f"s2|{model}|lr1{lr_str(s['lr1'])}|lr{lr_str(s['lr2'])}|"))
            if s["lr2"] is not None else t1)
        f = lambda x: x["tail_MRR"]
        print(f"{model:9s} {ms(nv, f):>22s} {ms(t1, f):>22s} {ms(t2, f):>22s}")
    if WS in sel:
        w = sel[WS]
        ws = coll(lambda k: k.startswith(
            f"ws|{WS}|lr1{lr_str(w['lr1'])}|emb{lr_str(w['emb'])}"
            f"|lr{lr_str(w['gnn'])}|"))
        print(f"{WS:9s} {'':>22s} {'':>22s} {ms(ws, lambda x: x['tail_MRR']):>22s}")
        # paired comparison with its control, RotatE stage 2
        s = sel["rotate"]
        ctrl = {k.rsplit("|", 1)[1]: v for k, v in R.items() if k.startswith(
            f"s2|rotate|lr1{lr_str(s['lr1'])}|lr{lr_str(s['lr2'])}|")}
        pairs = [(v["tail_MRR"], ctrl[k.rsplit("|", 1)[1]]["tail_MRR"])
                 for k, v in R.items() if k.startswith(
                     f"ws|{WS}|lr1{lr_str(w['lr1'])}|emb{lr_str(w['emb'])}"
                     f"|lr{lr_str(w['gnn'])}|")
                 and k.rsplit("|", 1)[1] in ctrl]
        if len(pairs) >= 2:
            d = np.array([a - b for a, b in pairs])
            line = (f"\n{WS} minus RotatE s2 (paired, n={len(d)}): "
                    f"{d.mean():+.4f} +/- {d.std(ddof=1):.4f}")
            try:
                from scipy import stats
                line += f"   paired t-test p = {stats.ttest_rel(*zip(*pairs)).pvalue:.3g}"
            except ImportError:
                pass
            print(line)

    print("\n== per pattern, two-stage (ws: warm-started), tail-only MRR ==")
    print(f"{'model':9s}" + "".join(f"{p[:9]:>10s}" for p in PATS))
    rows = []
    for model in MODELS:
        if model in sel:
            s = sel[model]
            pref = (f"s2|{model}|lr1{lr_str(s['lr1'])}|lr{lr_str(s['lr2'])}|"
                    if s["lr2"] is not None
                    else f"s1|{model}|lr{lr_str(s['lr1'])}|")
            rows.append((model, coll(lambda k: k.startswith(pref))))
    if WS in sel:
        w = sel[WS]
        rows.append((WS, coll(lambda k: k.startswith(
            f"ws|{WS}|lr1{lr_str(w['lr1'])}|emb{lr_str(w['emb'])}"
            f"|lr{lr_str(w['gnn'])}|"))))
    for model, xs in rows:
        line = f"{model:9s}"
        for p in PATS:
            v = [x["pattern_tail"].get(p) for x in xs
                 if x["pattern_tail"].get(p) is not None]
            line += f"{np.mean(v):10.3f}" if v else f"{'-':>10s}"
        print(line)
    flagged = [(k, v["test"].get("tied_frac", 0), v["test"].get("nan_frac", 0),
                v.get("diverged_epoch"))
               for k, v in R.items() if isinstance(v, dict) and "test" in v
               and (v["test"].get("tied_frac", 0) > 0.01
                    or v["test"].get("nan_frac", 0) > 0
                    or v.get("diverged_epoch") is not None)]
    print(f"\n== diagnostics: {len(flagged)} run(s) diverged, collapsed (>1% "
          f"tied gold scores) or produced NaN; none of them can be selected ==")
    for k, t, n, d in sorted(flagged):
        print(f"  {k:46s} tied {t:6.1%}  NaN {n:6.1%}"
              + (f"  diverged at epoch {d}" if d is not None else ""))
    meta = R.get("_meta")
    if meta:
        print(f"\nrun on {meta.get('device')} | torch {meta.get('torch')} | "
              f"data {meta.get('data_sha')[:12]} | code {meta.get('code_sha')[:12]}")
        print(f"protocol: d={meta.get('dim')}  margin={meta.get('margin')}  "
              f"batch={meta.get('bs')}  epochs<={meta.get('epochs')}  "
              f"metric={meta.get('metric')}")
        n = int(meta.get("n_ent", 0))
        if n:
            rnd = sum(1.0 / k for k in range(1, n + 1)) / n
            print(f"\nRandom-ranking MRR (exact): {rnd:.5f} for {n:,} "
                  f"candidates")


# ---- main --------------------------------------------------------------------
def _set_protocol(bs, dim, margin):
    global BS, DIM
    BS, DIM = int(bs), int(dim)
    models.MARGIN = float(margin)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", choices=["smoke", "sweep", "final", "report"])
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--no-ws", action="store_true",
                    help="skip the warm-started encoder")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates-from", default=None,
                    help="final only: reuse the rates selected in another "
                         "results file instead of sweeping this dataset")
    ap.add_argument("--allow-change", action="store_true",
                    help="do not use unless instructed: continue a results "
                         "file although data, code or device changed")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="epoch limit (default 160; the extended runs "
                         "of the paper use 640)")
    ap.add_argument("--bs", type=int, default=BS,
                    help="queries per mini-batch (default 512)")
    ap.add_argument("--dim", type=int, default=DIM,
                    help="embedding dimension d (default 64)")
    ap.add_argument("--margin", type=float, default=models.MARGIN,
                    help="margin of TransE, RotatE and WS-RGCN (default 9)")
    args = ap.parse_args()
    args.models_ws = [] if args.no_ws else [WS]
    _set_protocol(args.bs, args.dim, args.margin)

    if args.cmd == "report":
        if not args.out or not os.path.exists(args.out):
            raise SystemExit("--out must name an existing results file")
        args.data = args.data or None
        ctx = Ctx.__new__(Ctx)
        ctx.R = json.load(open(args.out))
        return cmd_report(ctx)

    if not args.data:
        raise SystemExit("--data is required")
    if args.cmd in ("sweep", "final") and not args.out:
        raise SystemExit("--out is required")
    _install_hooks()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() "
                             "is False")
        cap = torch.cuda.get_device_capability(0)
        print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  "
              f"{torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
    else:
        torch.set_num_threads(1)
        print(f"torch {torch.__version__}  CPU, 1 thread")
    print(f"protocol: d={DIM}  margin={models.MARGIN:g}  batch={BS}  "
          f"epochs<={args.epochs}  metric={METRIC}")
    ctx = Ctx(args)
    selftest(args.data, args.device)
    if args.out:
        check_fingerprint(ctx, args.allow_change)
    print(f"data {args.data}  ->  {args.out}  "
          f"({len(ctx.R)} records present)\n", flush=True)
    {"smoke": cmd_smoke, "sweep": cmd_sweep, "final": cmd_final}[args.cmd](ctx)


if __name__ == "__main__":
    main()
