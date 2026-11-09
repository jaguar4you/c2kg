# C2KG: a leakage-audited command-and-control knowledge graph benchmark

C2KG is a synthetic knowledge graph for link prediction in the
command-and-control (C2) domain. Its schema is organised around five of
the METT-TC mission variables (mission; enemy; terrain and weather; troops
and support available; time available), and its generator produces
symmetric, antisymmetric, inverse, composition and many-to-many relations
whose leakage between training and test sets is audited and controlled.
C2KG is a controlled benchmark informed by C2 concepts; it does not
reproduce the complexity, uncertainty or constraints of operational C2
data.

This repository contains the generator, the leakage audit, the
leakage-aware split, the training and evaluation code, the datasets and
the result files of the paper

> J. Cho, "C2KG: A Leakage-Audited Command-and-Control Knowledge Graph
> Benchmark and Three Confounds in Link-Prediction Evaluation", submitted
> for publication, 2026.

## Repository layout

```
c2kg/
├── code/                    generator, audit, split, models, runners, checks
├── data/
│   ├── c2kg_L/              C2KG-L, leakage-aware split (recommended), graph seed 42
│   ├── c2kg_N/              C2KG-N, inverse-pair split (for comparison), graph seed 42
│   ├── c2kg_N_blocked/      C2KG-N with every training triple that restates a test triple removed
│   ├── c2kg_L_g43/ ...      the two splits for graph seeds 43 and 44
│   ├── wn18rr_original/     standard WN18RR files (see data/LICENSE-WN18RR.txt)
│   ├── wn18rr_O/            WN18RR, original split, in the C2KG format
│   ├── wn18rr_O_blocked/    WN18RR, original split without the training mirrors of test triples
│   ├── wn18rr_L/            WN18RR, leakage-aware re-split with the original sizes
│   └── FINGERPRINTS.json    data fingerprints (see below)
└── results/                 result files of the paper
```

## The benchmark

| | C2KG (graph seed 42) |
|---|---|
| Entities / relations / triples | 1923 / 23 / 12,573 |
| Train / validation / test | 10,059 / 1257 / 1257 |
| Entity types | 10 (mission, unit, commander, equipment, location, terrain, weather, time, engagement, movement) |

Splits of the same graph:

* **C2KG-L** (recommended). A union-find groups every triple with its
  inverse-pair counterpart, its symmetric mirror and its co-referent copy
  (the same entity pair under another relation), and whole groups are
  assigned to one split. No test triple can be recovered by a one-hop
  lookup. Composition is kept as a measured inference pattern: 114 of the
  174 test triples of `nearLocation` (65.5 %) follow from
  `locatedAt` ∘ `adjacentTo` in the training set.
* **C2KG-N**. Only explicit inverse pairs are grouped. 248 of the 1257 test
  triples (19.7 %) have a symmetric mirror or co-referent copy in training.
  It is provided to measure the effect of such redundancy, not as a
  benchmark.
* **C2KG-N-blocked**. The validation and test sets of C2KG-N, and its
  training set without the 248 triples that restate a test triple (written
  to `filter_extra.txt`, which is used only for filtering). Comparing a
  model on C2KG-N and on this variant measures the effect of direct
  recoverability on exactly the same test triples
  (`code/make_blocked.py`).

All splits keep every entity and relation in the training set
(transductive setting). Further instances are drawn by changing the seed of
the generator; seeds 43 and 44 are included.

### File format

| File | Content |
|---|---|
| `train.txt`, `valid.txt`, `test.txt` | one triple per line: `head<TAB>relation<TAB>tail` (names) |
| `entities.dict`, `relations.dict` | `id<TAB>name` |
| `relation_patterns.json` | relation → pattern label (symmetric, antisymmetric, inverse, composition, N-N, general) |
| `split_report.json` | structures detected and grouped by the leakage-aware split (C2KG-L and WN18RR folders) |
| `filter_extra.txt`, `blocked_report.json` | removed triples and report (C2KG-N-blocked only) |

### Data fingerprints

Every results file stores a SHA-256 fingerprint of the data it was
computed on (`data_sha`: names and contents of the data files, line
endings normalised). `data/FINGERPRINTS.json` lists the fingerprints of the
folders in this repository; check them with

```
cd code
python check_data.py
```

| Dataset | `data_sha` (first 12 hex digits) |
|---|---|
| c2kg_L | `c64839834e08` |
| c2kg_N | `5d3f9bb15156` |
| c2kg_N_blocked | `cd539bf9c13b` |
| c2kg_L_g43 | `695f0696f57c` |
| c2kg_N_g43 | `2fbef686a58d` |
| c2kg_L_g44 | `f9ca8b0accc5` |
| c2kg_N_g44 | `c6f4c72a54c7` |
| wn18rr_O | `ca22fabe2697` |
| wn18rr_O_blocked | `d0d4c7bff14e` |
| wn18rr_L | `251af9c1c1b2` |

## Installation

Python 3.10 or later with PyTorch and NumPy; SciPy is optional (paired
t-test in the reports).

```
pip install -r requirements.txt
```

The experiments of the paper were run with PyTorch 2.11 and CUDA 12.8 on
one NVIDIA GeForce RTX 5090. Everything also runs on CPU (`--device cpu`),
more slowly. Floating-point results depend on the device and library
versions, so reruns elsewhere are close to, but not bitwise identical
with, the stored results; for the same reason the runner refuses to
continue a results file on a different device.

## Protocol

| Setting | Value |
|---|---|
| Embedding dimension | 64 (`--dim`) |
| Margin of TransE, RotatE, WS-RGCN | 9 (`--margin`) |
| Loss | 1-N (KvsAll) binary cross-entropy, label smoothing 0.1 |
| Optimiser | Adam, batch 512 queries (`--bs`), gradient-norm clipping at 5 |
| Epochs | at most 160 (`--epochs`); validation every 15 epochs, patience 6 |
| Learning rates | ladder {1, 3} × 10^k from 10^-4 to 3 × 10^1, edge extension, selected on seed-0 validation MRR |
| Stage 2 | continuation from the best stage-1 model with a fresh optimiser at its own rate ("no stage 2" competes) |
| Metrics | filtered ranking; MRR and Hits@k are exact expectations under uniformly random tie-breaking, MR the expected rank |

Ties: for a query with g candidates scored above the gold entity and m
tied with it (gold included), the expected reciprocal rank is
(1/m) Σ_{j=1..m} 1/(g+j) and the expected Hits@k is clip(k − g, 0, m)/m.
Every results file stores g and m of every test query, so any rank-based
metric can be recomputed exactly.

## Reproducing the paper

All commands are run from `code/`; they work unchanged in a Linux shell and
in Windows PowerShell. First the checks:

```
python test_metrics.py
python check_data.py
python c2kg_runner.py smoke --data ../data/c2kg_L
```

Regenerating the data (optional; the regenerated files are byte-identical
to those in `data/`):

```
python data_gen.py    --out ../data/c2kg_N --seed 42 --split-mode grouped
python split_fixed.py --out ../data/c2kg_L --seed 42
python make_blocked.py --src ../data/c2kg_N --out ../data/c2kg_N_blocked
python wn18rr_prepare.py --src ../data/wn18rr_original --out-root ../data
python make_blocked.py --src ../data/wn18rr_O --out ../data/wn18rr_O_blocked
```

Audit and training-free baselines:

```
python leakage_audit.py --data ../data/c2kg_L --rule locatedAt,adjacentTo,nearLocation
python path_probe.py --data ../data/c2kg_L --probe all --rule locatedAt,adjacentTo,nearLocation --out ../runs/probe_L.json
```

Trained models (`sweep` selects the learning rates on seed 0, `final`
trains all seeds at the selected rates, `report` prints the tables):

```
python c2kg_runner.py sweep  --data ../data/c2kg_L --out ../runs/c2kg_L.json
python c2kg_runner.py final  --data ../data/c2kg_L --out ../runs/c2kg_L.json
python c2kg_runner.py report --out ../runs/c2kg_L.json
```

The other experiments reuse the rates selected on graph 42 with
`--rates-from`, or change one setting (`--bs`, `--dim`, `--margin`,
`--epochs`) and select the rates again; `RUNBOOK.md` lists every command.
WN18RR uses `wn18rr_runner.py` with the same subcommands. Results are
written to `runs/`, which is not part of the repository. An interrupted
command resumes when it is run again.

## Code

| File | Purpose |
|---|---|
| `data_gen.py` | graph generator and inverse-pair split (C2KG-N) |
| `split_fixed.py` | leakage-aware split (C2KG-L) |
| `make_blocked.py` | leak-blocked variant of a split |
| `leakage_audit.py` | counts inverse, mirror, co-referent and composition leakage of a split |
| `path_probe.py` | training-free baselines: path rule, rule miner, frequency |
| `ontology.py` | entity types and relation domains and ranges |
| `models.py` | TransE, RotatE, DistMult, ComplEx, R-GCN, CompGCN |
| `warmstart_gnn.py` | WS-RGCN (`re_rgcn` in the results files): residual R-GCN warm-started from RotatE |
| `train_eval.py` | KvsAll training and filtered evaluation |
| `c2kg_runner.py` | rate selection, stage 1, stage 2, warm start, reports, safeguards |
| `wn18rr_prepare.py`, `wn18rr_runner.py` | WN18RR datasets and runs |
| `test_metrics.py` | checks the metrics against a Monte Carlo simulation of random tie-breaking |
| `check_data.py` | checks the data fingerprints |

## Changes from v1.0

* `supports` is generated strictly antisymmetric (v1.0 had 4 of 244 pairs in
  both directions); this changes the graphs of seeds 42 and 44.
* MRR and Hits@k are exact expectations under random tie-breaking (v1.0
  used the reciprocal of the expected rank).
* Batch size 512 (v1.0: 4096); selectable with `--bs`, as are `--dim` and
  `--margin`, all recorded in the results files.
* Degenerate runs are detected on the validation set; the test set enters
  no decision.
* New: C2KG-N-blocked and WN18RR-O-blocked, per-query g and m in the results files,
  `test_metrics.py`, `check_data.py`.

## License

* Code: MIT License (`LICENSE`).
* C2KG data (`data/c2kg_*`): Creative Commons Attribution 4.0
  International (`data/LICENSE-DATA.txt`).
* WN18RR files (`data/wn18rr_*`): derived from WordNet 3.0 and distributed
  under the WordNet License (`data/LICENSE-WN18RR.txt`).

## Citation

If you use C2KG, please cite the paper above (see `CITATION.cff`).
