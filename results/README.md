# Results

Every file here is produced by a command in `RUNBOOK.md`; all commands are
run from `code/`.

## Trained models (`runs/`, gzip-compressed)

Produced with release v1.1.0 on one NVIDIA GeForce RTX 5090 (PyTorch 2.11,
CUDA 12.8). `python unpack_results.py` unpacks them into `runs/`.

| File | Experiment |
|---|---|
| `c2kg_L.json.gz` | C2KG-L, graph 42: rate selection and five seeds of every model |
| `c2kg_N.json.gz` | C2KG-N, graph 42: rate selection and five seeds |
| `c2kg_Nb.json.gz` | C2KG-N-blocked, rates of C2KG-N, five seeds |
| `c2kg_L_g43.json.gz`, `c2kg_L_g44.json.gz` | C2KG-L, graphs 43 and 44, rates of graph 42, three seeds |
| `c2kg_N_g43.json.gz`, `c2kg_N_g44.json.gz` | C2KG-N, graphs 43 and 44, rates of graph 42, three seeds |
| `ext_L.json.gz` | TransE and RotatE on C2KG-L with 640 epochs |
| `ws_active_L.json.gz` | WS-RGCN on C2KG-L with the encoder rate fixed at 3e-4 (`rates_L_ws_active.json`) |
| `sens_bs4096_L.json.gz` | C2KG-L, batch 4096, rates selected again, three seeds |
| `sens_d128_L.json.gz` | C2KG-L, d = 128, rates selected again, three seeds |
| `sens_m6_L.json.gz`, `sens_m12_L.json.gz` | TransE and RotatE on C2KG-L, margin 6 and 12, three seeds |
| `wn_O.json.gz` | WN18RR, original split: rate selection and three seeds of RotatE, DistMult, ComplEx |
| `wn_Ob.json.gz` | WN18RR-O-blocked, rates of WN18RR-O, three seeds |
| `wn_L.json.gz` | WN18RR, leakage-aware re-split: rate selection and three seeds |

## Training-free analyses (release v1.1.1)

| File | Command |
|---|---|
| `schema_check.txt` | `python check_schema.py` |
| `audit_L.txt` (and `_N`, `_N_blocked`, `_L_g43`, `_N_g43`, `_L_g44`, `_N_g44`) | `python leakage_audit.py --data ../data/c2kg_L --rule locatedAt,adjacentTo,nearLocation` |
| `audit_wn_O.txt` (and `_O_blocked`, `_L`) | `python leakage_audit.py --data ../data/wn18rr_O` |
| `audit_threshold_sensitivity.txt` | `python audit_sensitivity.py` |
| `probe_L.json`, `probe_L.txt` (and the other C2KG folders) | `python path_probe.py --data ../data/c2kg_L --probe all --rule locatedAt,adjacentTo,nearLocation --out ../results/probe_L.json` |
| `probe_wn_O.json`, `probe_wn_O.txt` (and `_O_blocked`, `_L`) | `python path_probe.py --data ../data/wn18rr_O --probe all --out ../results/probe_wn_O.json` |
| `verification_v1.1.txt`, `verification_v1.1.1.txt` | checks of the code before the runs and of release v1.1.1 |
