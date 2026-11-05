# C2KG: a leakage-audited command-and-control knowledge graph benchmark

C2KG is a synthetic knowledge graph for link prediction in the
command-and-control (C2) domain. Its ontology is organised around five of
the METT-TC mission variables (mission; enemy; terrain and weather; troops
and support available; time available), and its generator injects symmetric,
antisymmetric, inverse, composition and many-to-many relations whose
leakage between training and test sets is audited and controlled.

This repository contains the generator, the leakage audit, the
leakage-aware split, the training and evaluation code, the datasets and
every result file used in the paper

> J. Cho, "C2KG: A Leakage-Audited Command-and-Control Knowledge Graph
> Benchmark and Three Confounds in Link-Prediction Evaluation", submitted
> for publication, 2026.

## Repository layout

```
c2kg/
├── code/                    generator, audit, split, models, runners
├── data/
│   ├── c2kg\_L/              C2KG-L, leakage-aware split (recommended), graph seed 42
│   ├── c2kg\_N/              C2KG-N, inverse-pair split (for comparison), graph seed 42
│   ├── c2kg\_L\_g43/ ...      the same two splits for graph seeds 43 and 44
│   ├── wn18rr\_original/     standard WN18RR files (see data/LICENSE-WN18RR.txt)
│   ├── wn18rr\_O/            WN18RR, original split, in the C2KG format
│   └── wn18rr\_L/            WN18RR, leakage-aware re-split with the original sizes
└── results/                 result files of the paper (JSON), audits, probes, reports
```

## The benchmark

||C2KG (graph seed 42)|
|-|-|
|Entities / relations / triples|1923 / 23 / 12,573|
|Train / validation / test|10,059 / 1257 / 1257|
|Entity types|10 (mission, unit, commander, equipment, location, terrain, weather, time, engagement, movement)|

Two splits of the same graph are provided.

* **C2KG-L** (recommended). A union-find groups every triple with its
inverse-pair counterpart, its symmetric mirror and its co-referent copy
(same entity pair under another relation), and whole groups are assigned
to one split. No test triple can be recovered by a one-hop lookup.
Composition is kept as a measured inference pattern: 127 of the 192 test
triples of `nearLocation` (66.1 %) follow from `locatedAt` ∘ `adjacentTo`
in the training set.
* **C2KG-N**. Only explicit inverse pairs are grouped. 247 of the 1257 test
triples (19.6 %) have a symmetric mirror or co-referent copy in training.
It is provided to measure the effect of leakage, not as a benchmark.

Both splits keep every entity and relation in the training set
(transductive setting). Further instances are drawn by changing the seed of
the generator; seeds 43 and 44 are included.

### File format

Each data folder contains

|File|Content|
|-|-|
|`train.txt`, `valid.txt`, `test.txt`|one triple per line: `head<TAB>relation<TAB>tail` (names)|
|`entities.dict`, `relations.dict`|`id<TAB>name`|
|`relation\_patterns.json`|relation → pattern label (symmetric, antisymmetric, inverse, composition, N-N, general)|
|`split\_report.json`|structures detected and grouped by the leakage-aware split (C2KG-L and WN18RR folders)|

### Data fingerprints

`code/c2kg\_runner.py` computes a SHA-256 fingerprint of each dataset
(`data\_sha`: file names and contents of the five data files and
`relation\_patterns.json`, line endings normalised). Every result file
stores the fingerprint of the data it was computed on.

|Dataset|`data\_sha` (first 12 hex digits)|
|-|-|
|c2kg\_L|`80ac24a167e9`|
|c2kg\_N|`074d5bd352ca`|
|c2kg\_L\_g43|`695f0696f57c`|
|c2kg\_N\_g43|`2fbef686a58d`|
|c2kg\_L\_g44|`a28f175080e2`|
|c2kg\_N\_g44|`15656cc2fd0d`|
|wn18rr\_O|`ca22fabe2697`|
|wn18rr\_L|`251af9c1c1b2`|

## Installation

Python 3.10 or later with PyTorch and NumPy; SciPy is optional (paired
t-test in the reports).

```
pip install -r requirements.txt
```

The experiments of the paper were run with PyTorch 2.11 and CUDA 12.8 on
one NVIDIA GeForce RTX 5090. Everything also runs on CPU (`--device cpu`),
more slowly. Floating-point results depend on the device and library
versions, so reruns elsewhere are close to, but not bitwise identical with,
the stored results; for the same reason the runner refuses to continue a
results file on a different device.

## Reproducing the paper

All commands are run from `code/`. They work unchanged in a Linux shell and
in Windows PowerShell.

### 1\. Regenerate the data (optional; the data are included)

```
python data\_gen.py    --out ../data/c2kg\_N --seed 42 --split-mode grouped
python split\_fixed.py --out ../data/c2kg\_L --seed 42
python wn18rr\_prepare.py --src ../data/wn18rr\_original --out-root ../data
```

Use seeds 43 and 44 (and folder names `c2kg\_N\_g43`, `c2kg\_L\_g43`, ...) for
the other instances. The regenerated files are byte-identical to the ones
in `data/` and reproduce the fingerprints above.

### 2\. Leakage audit and training-free baselines

```
python leakage\_audit.py --data ../data/c2kg\_L --rule locatedAt,adjacentTo,nearLocation
python leakage\_audit.py --data ../data/wn18rr\_O
python path\_probe.py --data ../data/c2kg\_L --probe all --rule locatedAt,adjacentTo,nearLocation --out ../runs/probe\_L.json
```

### 3\. Trained models

```
python c2kg\_runner.py smoke  --data ../data/c2kg\_L
python c2kg\_runner.py sweep  --data ../data/c2kg\_L --out ../runs/c2kg\_L.json
python c2kg\_runner.py final  --data ../data/c2kg\_L --out ../runs/c2kg\_L.json
python c2kg\_runner.py report --out ../runs/c2kg\_L.json
```

`sweep` selects the learning rates on seed 0 (validation MRR), `final`
trains five seeds at the selected rates, `report` prints the tables. The
same commands with `c2kg\_N` give C2KG-N. For the other graph instances,
reuse the rates selected on graph 42:

```
python c2kg\_runner.py final --data ../data/c2kg\_L\_g43 --out ../runs/c2kg\_L\_g43.json --rates-from ../runs/c2kg\_L.json --seeds 0 1 2
```

The runs with a 640-epoch limit (TransE and RotatE, C2KG-L):

```
python c2kg\_runner.py final --data ../data/c2kg\_L --out ../runs/ext\_L.json --rates-from ../runs/c2kg\_L.json --models transe rotate --no-ws --epochs 640
```

For WN18RR use `wn18rr\_runner.py` with the same four subcommands
(`--data ../data/wn18rr\_O` or `../data/wn18rr\_L`). Results are written to
`runs/`, which is not part of the repository, so that the files in
`results/` are never overwritten. An interrupted command resumes when it is
run again.

### Protocol

|Setting|Value|
|-|-|
|Embedding dimension|64|
|Loss|1-N (KvsAll) binary cross-entropy, label smoothing 0.1|
|Optimiser|Adam, batch 4096 queries (WN18RR: 512)|
|Epochs|at most 160; validation every 15 epochs, patience 6|
|Learning rates|ladder {1, 3} × 10^k from 10^-4 to 3 × 10^1, edge extension, selected on seed-0 validation MRR|
|Stage 2|continuation from the final stage-1 model with a fresh optimiser at its own rate ("no stage 2" competes)|
|Evaluation|filtered ranking, ties at the expected rank; primary metric: tail MRR|

## Reference results (C2KG-L, tail queries, five seeds)

|Model|MRR|Hits@1|Hits@3|Hits@10|MRR (head and tail)|
|-|-|-|-|-|-|
|TransE|0.078 ± 0.002|0.018|0.073|0.195|0.056 ± 0.001|
|RotatE|0.213 ± 0.003|0.155|0.212|0.329|0.192 ± 0.002|
|DistMult|0.042 ± 0.007|0.017|0.038|0.085|0.036 ± 0.007|
|ComplEx|0.050 ± 0.010|0.020|0.051|0.106|0.047 ± 0.010|
|R-GCN|0.049 ± 0.004|0.011|0.037|0.116|0.032 ± 0.006|
|CompGCN|0.051 ± 0.003|0.014|0.040|0.115|0.033 ± 0.002|
|WS-RGCN|0.212 ± 0.005|0.154|0.214|0.325|0.192 ± 0.006|
|Path rule|0.102|0.101|–|0.101|–|
|Frequency|0.045|0.006|–|0.116|–|

## Result files

|File(s) in `results/`|Content|Paper|
|-|-|-|
|`audit\_L.txt`, `audit\_N.txt`|leakage audit of the two splits|Table 3|
|`probe\_L.json/.txt`, `probe\_N.json/.txt`|path rule, rule miner, frequency baseline|Tables 7, 10, 11|
|`c2kg\_L.json`, `c2kg\_N.json`|rate sweeps and five-seed runs of all models except CompGCN|Tables 7–11|
|`cg\_L.json`, `cg\_N.json`|the same for CompGCN|Tables 7, 8, 10, 11|
|`ext\_L.json`|TransE and RotatE with the epoch limit raised to 640|Sections 4.3, 4.4; Table 9|
|`wn\_O.json`, `wn\_L.json`, `audit\_wn\_O.txt`, `audit\_wn\_L.txt`|WN18RR, original split and re-split|Table 12|
|`c2kg\_{L,N}\_g43.json`, `c2kg\_{L,N}\_g44.json`, `cg\_{L,N}\_g43.json`, `cg\_{L,N}\_g44.json`|graph instances 43 and 44|Table 13|
|`reports/`|output of the `report` subcommand for every JSON file||

Notes on the JSON files:

* Keys: `s1|<model>|lr<η1>|s<seed>` (stage 1), `s2|<model>|lr1<η1>|lr<η2>|s<seed>`
(stage 2), `ws|re\_rgcn|lr1<η1>|emb<η>|lr<η>|s<seed>` (warm start), `selected`
(selected rates), `\_meta` (data and code fingerprints, device, library
versions). The shared-rate runs are the stage-1 runs at `lr1e-2`.
* `tail\_MRR` is the primary metric; `test\_MRR` averages head and tail
queries; `val\_mrr` (head and tail, validation set) is the selection
criterion; `pattern\_tail` breaks the tail MRR down by relation pattern.
* `re\_rgcn` is the model called WS-RGCN in the paper.
* The `compgcn` records in `c2kg\_\*.json` come from an earlier CompGCN
encoder without residual connections and are not used; all CompGCN
results of the paper come from `cg\_\*.json`.
* The files also contain type-constrained metrics (`constrained`), in which
candidates violating the ontology are excluded; the paper does not use
them.
* The code in this repository was tidied after the experiments (comments,
removal of unused model variants), so its code fingerprint differs from
the one stored in the result files. Training and evaluation are
unchanged: the data generated by this code are byte-identical, and short
runs of every model and protocol give results identical to the last
digit with both versions. To rerun an experiment, write to a new results
file (as in the commands above); continuing a file of `results/` stops
with a fingerprint mismatch, by design.

## Code

|File|Purpose|
|-|-|
|`data\_gen.py`|graph generator and inverse-pair split (C2KG-N)|
|`split\_fixed.py`|leakage-aware split (C2KG-L)|
|`leakage\_audit.py`|counts inverse, mirror, co-referent and composition leakage of a split|
|`path\_probe.py`|training-free baselines: path rule, rule miner, frequency|
|`ontology.py`|entity types and relation domains and ranges|
|`models.py`|TransE, RotatE, DistMult, ComplEx, R-GCN, CompGCN|
|`warmstart\_gnn.py`|WS-RGCN: residual R-GCN warm-started from RotatE|
|`train\_eval.py`|KvsAll training and filtered evaluation|
|`c2kg\_runner.py`|rate selection, stage 1, stage 2, warm start, reports, safeguards|
|`wn18rr\_prepare.py`, `wn18rr\_runner.py`|WN18RR datasets and runs|

## License

* Code: MIT License (`LICENSE`).
* C2KG data (`data/c2kg\_\*`): Creative Commons Attribution 4.0
International (`data/LICENSE-DATA.txt`).
* WN18RR files (`data/wn18rr\_\*`): derived from WordNet 3.0 and distributed
under the WordNet License (`data/LICENSE-WN18RR.txt`).

## Citation

If you use C2KG, please cite the paper above (see `CITATION.cff`).

