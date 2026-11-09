"""
wn18rr_runner.py
----------------
WN18RR experiment: does the leakage-aware split change conclusions on WN18RR?

Runs RotatE, ComplEx and DistMult on splits of the same WN18RR triples
(prepared by wn18rr_prepare.py):

  wn18rr_O   the original split (34.7 % of its test triples have their
             symmetric mirror in training)
  wn18rr_L   the leakage-aware re-split with the original sizes
  wn18rr_O_blocked  the original split without the training triples that
             restate a test triple (make_blocked.py); final runs only, with
             --rates-from the results file of wn18rr_O

Everything is done by c2kg_runner.py, unchanged, so that the protocol is
the one used on C2KG: d = 64, KvsAll BCE with label smoothing 0.1, Adam,
batch 512, at most 160 epochs, validation every 15, patience 6, learning
rates from the same ladder selected on seed-0 validation MRR with automatic
edge extension, stage 1 and stage 2 ("no stage 2" competing), filtered
ranking with exact expectations under random tie-breaking.  This wrapper
changes only what has to differ for WN18RR:

  * models   RotatE, ComplEx, DistMult; no warm-started encoder
  * seeds    final runs use training seeds 0, 1, 2 by default
  * final    no shared-rate (lr 1e-2) runs; the learning-rate confound is
             studied on C2KG only (seed 0 at 1e-2 is part of the sweep anyway)
  * types    no ontology type masks (WN18RR has no entity types)
  * fingerprint  also hashes this file and wn18rr_prepare.py
  * report   adds the head-and-tail (two-direction) metrics usually reported
             for WN18RR

Usage (from the code folder, after wn18rr_prepare.py)
  python wn18rr_runner.py smoke  --data ../data/wn18rr_L --device cuda
  python wn18rr_runner.py sweep  --data ../data/wn18rr_O --out ../runs/wn_O.json --device cuda
  python wn18rr_runner.py final  --data ../data/wn18rr_O --out ../runs/wn_O.json --device cuda
  python wn18rr_runner.py report --out ../runs/wn_O.json
  (the same four with wn18rr_L and wn_L.json)
"""

import os
import sys
import time

import numpy as np

import c2kg_runner as C

WN_MODELS = ["rotate", "complex", "distmult"]
HERE = os.path.dirname(os.path.abspath(__file__))


# ---- what differs from C2KG ---------------------------------------------------
C.BS = 512          # the default of c2kg_runner.py as well; see the docstring
C.build_type_mask = lambda id2ent, rel2id: None          # no entity types


def code_sha():
    return C._sha([os.path.join(HERE, f) for f in
                   ("c2kg_runner.py", "train_eval.py", "models.py",
                    "warmstart_gnn.py", "ontology.py",
                    "wn18rr_runner.py", "wn18rr_prepare.py")])


C.code_sha = code_sha


def cmd_smoke(ctx):
    print("2-epoch run of every WN18RR model; numbers are meaningless here.")
    for model in ctx.args.models:
        t = time.time()
        res = C.train(ctx, model, 1e-2, 0, epochs=2)
        print(f"  {model:9s} ok  tail {res['test']['tail_only']['MRR']:.4f}"
              f"  {time.time() - t:6.1f}s", flush=True)
    print("SMOKE OK")


def cmd_final(ctx):
    src = ctx.args.rates_from
    if src:                          # e.g. the leak-blocked variant
        other = __import__("json").load(open(src))
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
            C.run_s1(ctx, model, s["lr1"], seed)
            if s["lr2"] is not None:
                C.run_s2(ctx, model, s["lr1"], s["lr2"], seed)
        print(f"  seed {seed} done ({ctx.elapsed()})", flush=True)
    print(f"\nFINAL DONE ({ctx.elapsed()})")


def cmd_report(ctx):
    _base_report(ctx)

    # head-and-tail metrics, as usually reported for WN18RR
    R, sel = ctx.R, ctx.R.get("selected", {})
    print("\n== final: head-and-tail and tail-only test metrics "
          "(selected protocol), mean +/- std over seeds ==")
    print(f"{'model':9s} {'stage':6s} {'n':>2s} {'MRR':>15s} {'H@1':>15s} "
          f"{'H@10':>15s} | {'tail MRR':>15s} {'tail H@10':>15s}")
    for model in WN_MODELS:
        if model not in sel:
            continue
        s = sel[model]
        if s["lr2"] is not None:
            pref, stage = (f"s2|{model}|lr1{C.lr_str(s['lr1'])}"
                           f"|lr{C.lr_str(s['lr2'])}|"), "s2"
        else:
            pref, stage = f"s1|{model}|lr{C.lr_str(s['lr1'])}|", "s1"
        xs = [v for k, v in R.items() if k.startswith(pref)]

        def ms(f):
            a = np.array([f(x) for x in xs], dtype=float)
            if not len(a):
                return "-"
            sd = a.std(ddof=1) if len(a) > 1 else 0.0
            return f"{a.mean():.4f}+/-{sd:.4f}"
        print(f"{model:9s} {stage:6s} {len(xs):2d} "
              f"{ms(lambda x: x['test']['MRR']):>15s} "
              f"{ms(lambda x: x['test']['H@1']):>15s} "
              f"{ms(lambda x: x['test']['H@10']):>15s} | "
              f"{ms(lambda x: x['tail_MRR']):>15s} "
              f"{ms(lambda x: x['tail_H@10']):>15s}")


_base_report = C.cmd_report
C.cmd_smoke, C.cmd_final, C.cmd_report = cmd_smoke, cmd_final, cmd_report


# ---- argument defaults --------------------------------------------------------
def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("smoke", "sweep", "final", "report"):
        raise SystemExit("usage: python wn18rr_runner.py "
                         "{smoke,sweep,final,report} --data ... --out ...")
    if "--rates-from" in argv and argv[0] != "final":
        raise SystemExit("--rates-from applies to final only")
    if "--models" in argv:
        i = argv.index("--models") + 1
        j = i
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        bad = [m for m in argv[i:j] if m not in WN_MODELS]
        if bad:
            raise SystemExit(f"WN18RR models are {WN_MODELS}; got {bad}")
    else:
        argv += ["--models"] + WN_MODELS
    if "--no-ws" not in argv:
        argv.append("--no-ws")
    if argv[0] == "final" and "--seeds" not in argv:
        argv += ["--seeds", "0", "1", "2"]
    sys.argv = [sys.argv[0]] + argv
    C.main()


if __name__ == "__main__":
    main()
