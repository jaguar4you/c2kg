# Runbook of the v1.1 experiments

Every command is run from `code/`. Results go to `../runs/` (one JSON file
per experiment; model checkpoints of stage 1 in `../runs/ckpt/`). An
interrupted command resumes when it is started again with the same
arguments. A results file can only be continued with the same data, code,
device and settings; anything else stops with a fingerprint message.

The commands are grouped into three independent queues that can run at the
same time in three terminals. Inside a queue, the order matters: `final`
needs the rates selected by `sweep`, and `--rates-from` needs the results
file of graph 42.

## 0. Checks (a few minutes)

```
python test_metrics.py
python check_data.py
python c2kg_runner.py smoke --data ../data/c2kg_L --device cuda
python wn18rr_runner.py smoke --data ../data/wn18rr_L --device cuda
```

## Queue A: C2KG-L (graph 42), longer epoch budget, instances

```
python c2kg_runner.py sweep --data ../data/c2kg_L --out ../runs/c2kg_L.json --device cuda
python c2kg_runner.py final --data ../data/c2kg_L --out ../runs/c2kg_L.json --device cuda
python c2kg_runner.py final --data ../data/c2kg_L --out ../runs/ext_L.json --rates-from ../runs/c2kg_L.json --models transe rotate --no-ws --epochs 640 --device cuda
python c2kg_runner.py final --data ../data/c2kg_L_g43 --out ../runs/c2kg_L_g43.json --rates-from ../runs/c2kg_L.json --seeds 0 1 2 --device cuda
python c2kg_runner.py final --data ../data/c2kg_L_g44 --out ../runs/c2kg_L_g44.json --rates-from ../runs/c2kg_L.json --seeds 0 1 2 --device cuda
```

## Queue B: C2KG-N (graph 42), leak-blocked variant, instances

```
python c2kg_runner.py sweep --data ../data/c2kg_N --out ../runs/c2kg_N.json --device cuda
python c2kg_runner.py final --data ../data/c2kg_N --out ../runs/c2kg_N.json --device cuda
python c2kg_runner.py final --data ../data/c2kg_N_blocked --out ../runs/c2kg_Nb.json --rates-from ../runs/c2kg_N.json --device cuda
python c2kg_runner.py final --data ../data/c2kg_N_g43 --out ../runs/c2kg_N_g43.json --rates-from ../runs/c2kg_N.json --seeds 0 1 2 --device cuda
python c2kg_runner.py final --data ../data/c2kg_N_g44 --out ../runs/c2kg_N_g44.json --rates-from ../runs/c2kg_N.json --seeds 0 1 2 --device cuda
```

## Queue C: WN18RR, then sensitivity runs on C2KG-L

```
python wn18rr_runner.py sweep --data ../data/wn18rr_O --out ../runs/wn_O.json --device cuda
python wn18rr_runner.py final --data ../data/wn18rr_O --out ../runs/wn_O.json --device cuda
python wn18rr_runner.py final --data ../data/wn18rr_O_blocked --out ../runs/wn_Ob.json --rates-from ../runs/wn_O.json --device cuda
python wn18rr_runner.py sweep --data ../data/wn18rr_L --out ../runs/wn_L.json --device cuda
python wn18rr_runner.py final --data ../data/wn18rr_L --out ../runs/wn_L.json --device cuda

python c2kg_runner.py sweep --data ../data/c2kg_L --out ../runs/sens_bs4096_L.json --bs 4096 --device cuda
python c2kg_runner.py final --data ../data/c2kg_L --out ../runs/sens_bs4096_L.json --bs 4096 --seeds 0 1 2 --device cuda
python c2kg_runner.py sweep --data ../data/c2kg_L --out ../runs/sens_d128_L.json --dim 128 --no-ws --device cuda
python c2kg_runner.py final --data ../data/c2kg_L --out ../runs/sens_d128_L.json --dim 128 --no-ws --seeds 0 1 2 --device cuda
python c2kg_runner.py sweep --data ../data/c2kg_L --out ../runs/sens_m6_L.json --margin 6 --models transe rotate --no-ws --device cuda
python c2kg_runner.py final --data ../data/c2kg_L --out ../runs/sens_m6_L.json --margin 6 --models transe rotate --no-ws --seeds 0 1 2 --device cuda
python c2kg_runner.py sweep --data ../data/c2kg_L --out ../runs/sens_m12_L.json --margin 12 --models transe rotate --no-ws --device cuda
python c2kg_runner.py final --data ../data/c2kg_L --out ../runs/sens_m12_L.json --margin 12 --models transe rotate --no-ws --seeds 0 1 2 --device cuda
```

## Reports

```
python c2kg_runner.py report --out ../runs/c2kg_L.json
python wn18rr_runner.py report --out ../runs/wn_O.json
```

The audits, training-free baselines and threshold sensitivity of the audit
do not involve training and are in `results/` already.

## Analysis plan (fixed before the v1.1 runs)

* Primary metric: filtered tail MRR (exact expectation under random
  tie-breaking); head-and-tail MRR, Hits@{1,3,10} and type-constrained
  metrics are reported alongside. Rates are selected on the seed-0
  validation MRR over head and tail queries; the test set enters no
  decision.
* Effect of redundancy: C2KG-N against C2KG-N-blocked (and WN18RR-O against
  WN18RR-O-blocked) on the same test triples, separately for the test
  triples that have a restating training triple in the unblocked split and
  for the others; `nearLocation` (composition) triples are reported
  separately because blocking also removes some of their two-hop paths.
* Warm-started encoder: WS-RGCN against RotatE stage 2 (the continuation
  control), paired over training seeds. Equivalence is tested with two
  one-sided tests (TOST, alpha = 0.05) against a margin of +/- 0.01 tail
  MRR; across the three graph instances, effects are reported per graph
  and pooled with the graph as a cluster.
* Sensitivity (batch 4096, d = 128, margin 6 and 12, 640 epochs): whether
  the order of the models, the penalty of the shared rate and the gain of
  stage 2 persist.
