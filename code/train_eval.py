"""
train_eval.py
-------------
Data loading, adjacency construction, 1-vs-all (KvsAll) cross-entropy
training and filtered link-prediction evaluation (MRR, MR, Hits@{1,3,10})
with a per-relation-pattern breakdown.

Training objective
------------------
1-N (KvsAll) multi-label binary cross-entropy with label smoothing
(Dettmers et al. 2018; Ruffinelli et al. 2020).  Every pair (h, r) in the
training set forms a tail query whose targets are all of its true tails,
and every pair (r, t) a head query whose targets are all of its true heads.
Both are scored against all entities in one matrix product; head queries
are scored directly (score_all_h), not through added inverse relations.

Models with a graph encoder recompute the entity representations over the
full training graph for every mini-batch, so that they receive one
optimisation step per mini-batch, like the embedding models.
"""

import os, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from models import build_model


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_dicts(data_dir):
    ent2id, rel2id = {}, {}
    for line in open(os.path.join(data_dir, "entities.dict")):
        i, e = line.strip().split("\t"); ent2id[e] = int(i)
    for line in open(os.path.join(data_dir, "relations.dict")):
        i, r = line.strip().split("\t"); rel2id[r] = int(i)
    return ent2id, rel2id


def load_triples(path, ent2id, rel2id):
    out = []
    for line in open(path):
        h, r, t = line.strip().split("\t")
        out.append((ent2id[h], rel2id[r], ent2id[t]))
    return np.array(out, dtype=np.int64)


def build_adj(train, n_ent, n_rel, device):
    rows = defaultdict(list)
    for h, r, t in train:
        rows[r].append((t, h))
        rows[r + n_rel].append((h, t))
    deg = np.zeros((2 * n_rel, n_ent), dtype=np.float64)
    for r, lst in rows.items():
        for d, s in lst:
            deg[r, d] += 1.0
    adj_list = []
    for r, lst in rows.items():
        idx = torch.tensor(list(zip(*lst)), dtype=torch.long)
        d = deg[r, idx[0].numpy()]
        val = torch.tensor(1.0 / np.clip(d, 1.0, None), dtype=torch.float32)
        A = torch.sparse_coo_tensor(idx, val, (n_ent, n_ent)).coalesce()
        adj_list.append((r, A.to(device)))
    return adj_list


def all_true(*tripsets):
    t_filter = defaultdict(set)
    h_filter = defaultdict(set)
    for trips in tripsets:
        for h, r, t in trips:
            t_filter[(h, r)].add(t)
            h_filter[(r, t)].add(h)
    return t_filter, h_filter


def kvsall_queries(train):
    """Tail queries (h,r)->{t} and head queries (r,t)->{h}."""
    qt = defaultdict(set)
    qh = defaultdict(set)
    for h, r, t in train:
        qt[(h, r)].add(t)
        qh[(r, t)].add(h)
    return qt, qh


# ----------------------------------------------------------------------------
# Evaluation (filtered)
# ----------------------------------------------------------------------------
def _rank(scores, true):                           # [tie/NaN fix]
    """Rank of the gold entity with ties broken at random, in expectation.

    rank = 1 + #(score > gold) + (#(score == gold) - 1) / 2

    A strict-'>' rule would rank the gold first among every candidate
    tied with it.  A model whose representations collapse (e.g. dead ReLU
    units giving identical scores) then receives MRR close to 1.0.  The
    expected rank under random tie-breaking is the protocol recommended by
    Sun et al. (ACL 2020).  Without ties both rules give the same rank."""
    g = true.unsqueeze(1)
    greater = (scores > g).sum(1).double()
    equal = (scores == g).sum(1).double()          # includes the gold
    return greater + 1.0 + (equal - 1.0) / 2.0


def _no_nan(scores):                               # [tie/NaN fix]
    """NaN scores (a diverged model) rank last instead of first."""
    return torch.where(torch.isnan(scores),
                       torch.full_like(scores, float("-inf")), scores)


@torch.no_grad()
def evaluate(model, triples, E, t_filter, h_filter, n_ent, device,
             rel_pattern=None, id2rel=None, batch=512, type_masks=None):
    """Filtered link-prediction evaluation.

    If ``type_masks`` is given (dict relation_id -> (head_mask, tail_mask)
    of bool tensors of length n_ent), the function additionally computes
    type-constrained metrics in which ontology-invalid candidates receive
    score -inf.  This is computed alongside the
    unconstrained metrics in a single pass."""
    model.eval()
    n_q = n_tied = n_nan = 0                       # [tie/NaN fix]
    ranks, ranks_c = [], []
    ranks_t, ranks_t_c = [], []          # # [tail-only patch] tail-side only
    per_pat = defaultdict(list)
    per_pat_c = defaultdict(list)
    per_pat_t = defaultdict(list)        # # [tail-only patch]
    T = torch.tensor(triples, dtype=torch.long, device=device)
    if type_masks is not None:
        # pre-build per-relation mask tensors once
        tmask_t = {r_: torch.tensor(m[1], device=device) for r_, m in type_masks.items()}
        tmask_h = {r_: torch.tensor(m[0], device=device) for r_, m in type_masks.items()}

    for i in range(0, len(T), batch):
        b = T[i:i + batch]
        h, r, t = b[:, 0], b[:, 1], b[:, 2]
        ar = torch.arange(len(b), device=device)

        st = model.score_all_t(E, h, r)
        n_nan += int(torch.isnan(st).any(1).sum())     # [tie/NaN fix]
        st = _no_nan(st)
        true_t = st[ar, t].clone()
        for j in range(len(b)):
            st[j, list(t_filter[(int(h[j]), int(r[j]))])] = float("-inf")  # [guard]
        st[ar, t] = true_t

        # --- unconstrained tail rank
        rank_t = _rank(st, true_t)                       # [tie/NaN fix]
        n_tied += int(((st == true_t.unsqueeze(1)).sum(1) > 1).sum())

        # --- type-constrained tail rank
        rank_t_c = None
        if type_masks is not None:
            st_c = st.clone()
            for j in range(len(b)):
                m = ~tmask_t[int(r[j])]
                st_c[j, m] = float("-inf")  # [guard]
            st_c[ar, t] = true_t        # never mask the gold
            rank_t_c = _rank(st_c, true_t)

        sh = model.score_all_h(E, r, t)
        n_nan += int(torch.isnan(sh).any(1).sum())     # [tie/NaN fix]
        sh = _no_nan(sh)
        true_h = sh[ar, h].clone()
        for j in range(len(b)):
            sh[j, list(h_filter[(int(r[j]), int(t[j]))])] = float("-inf")  # [guard]
        sh[ar, h] = true_h

        rank_h = _rank(sh, true_h)                       # [tie/NaN fix]
        n_tied += int(((sh == true_h.unsqueeze(1)).sum(1) > 1).sum())
        n_q += 2 * len(b)

        rank_h_c = None
        if type_masks is not None:
            sh_c = sh.clone()
            for j in range(len(b)):
                m = ~tmask_h[int(r[j])]
                sh_c[j, m] = float("-inf")  # [guard]
            sh_c[ar, h] = true_h
            rank_h_c = _rank(sh_c, true_h)

        rt = rank_t.cpu().numpy(); rh = rank_h.cpu().numpy()
        rr = r.cpu().numpy()
        if type_masks is not None:
            rt_c = rank_t_c.cpu().numpy(); rh_c = rank_h_c.cpu().numpy()
        for j in range(len(b)):
            ranks.append(rt[j]); ranks.append(rh[j])
            ranks_t.append(rt[j])                    # # [tail-only patch]
            if type_masks is not None:
                ranks_c.append(rt_c[j]); ranks_c.append(rh_c[j])
                ranks_t_c.append(rt_c[j])            # # [tail-only patch]
            if rel_pattern is not None:
                p = rel_pattern[id2rel[int(rr[j])]]
                per_pat[p].append(rt[j]); per_pat[p].append(rh[j])
                per_pat_t[p].append(rt[j])           # # [tail-only patch]
                if type_masks is not None:
                    per_pat_c[p].append(rt_c[j]); per_pat_c[p].append(rh_c[j])

    def _agg(ranks_arr):
        ranks_arr = np.array(ranks_arr, dtype=np.float64)
        return {"MRR": float(np.mean(1.0 / ranks_arr)),
                "MR":  float(np.mean(ranks_arr)),
                "H@1": float(np.mean(ranks_arr <= 1)),
                "H@3": float(np.mean(ranks_arr <= 3)),
                "H@10": float(np.mean(ranks_arr <= 10))}

    res = _agg(ranks)
    res["tail_only"] = _agg(ranks_t)                 # # [tail-only patch]
    # share of queries (tail and head) whose gold score is tied with at
    # least one other candidate, and whose score row contains NaN
    res["tied_frac"] = n_tied / max(n_q, 1)          # [tie/NaN fix]
    res["nan_frac"] = n_nan / max(n_q, 1)
    pat_res = {p: {"MRR": float(np.mean(1.0 / np.array(rk))),
                   "H@10": float(np.mean(np.array(rk) <= 10)),
                   "n": int(len(rk))} for p, rk in per_pat.items()}
    for p, rk in per_pat_t.items():                  # # [tail-only patch]
        a = np.array(rk, dtype=np.float64)
        pat_res.setdefault(p, {})
        pat_res[p]["MRR_tail"] = float(np.mean(1.0 / a))
        pat_res[p]["H@10_tail"] = float(np.mean(a <= 10))
        pat_res[p]["n_tail"] = int(len(a))
    if type_masks is not None:
        res_c = _agg(ranks_c)
        pat_res_c = {p: {"MRR": float(np.mean(1.0 / np.array(rk))),
                         "H@10": float(np.mean(np.array(rk) <= 10)),
                         "n": int(len(rk))} for p, rk in per_pat_c.items()}
        res_c["tail_only"] = _agg(ranks_t_c)         # # [tail-only patch]
        res["constrained"] = res_c
        res["constrained_pattern"] = pat_res_c
    return res, pat_res


# ----------------------------------------------------------------------------
# Training (KvsAll cross-entropy)
# ----------------------------------------------------------------------------
def train_model(name, data_dir, dim=200, epochs=300, lr=2e-3,
                 label_smooth=0.1, reg=1e-7, num_bases=15, dropout=0.2,
                 seed=0, device="cpu",
                 eval_every=15, patience=8, bs=512, verbose=False,
                 ckpt_path=None, time_budget=None, type_masks=None):
    """Train with 1-vs-all CE.  If ``ckpt_path`` is given the training
    state is periodically persisted; when ``time_budget`` (seconds) is
    exceeded the function saves and returns ``{"status": "incomplete"}``
    so the caller can re-invoke and resume exactly where it left off."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    ent2id, rel2id = load_dicts(data_dir)
    id2rel = {v: k for k, v in rel2id.items()}
    n_ent, n_rel = len(ent2id), len(rel2id)
    train = load_triples(os.path.join(data_dir, "train.txt"), ent2id, rel2id)
    valid = load_triples(os.path.join(data_dir, "valid.txt"), ent2id, rel2id)
    test  = load_triples(os.path.join(data_dir, "test.txt"),  ent2id, rel2id)
    rel_pattern = json.load(open(os.path.join(data_dir,
                                              "relation_patterns.json")))
    t_filt, h_filt = all_true(train, valid, test)

    # bidirectional KvsAll queries: tail (h,r)->T and head (r,t)->H
    qt, qh = kvsall_queries(train)
    tail_keys = list(qt.keys())
    head_keys = list(qh.keys())
    # unified query table: [a, rel, direction]  dir 0 = tail, 1 = head
    rows, targets = [], []
    for (h, r) in tail_keys:
        rows.append((h, r, 0)); targets.append(sorted(qt[(h, r)]))
    for (r, t) in head_keys:
        rows.append((t, r, 1)); targets.append(sorted(qh[(r, t)]))
    Q = torch.tensor(rows, dtype=torch.long, device=device)
    qtgt = [torch.tensor(t, dtype=torch.long, device=device) for t in targets]

    model = build_model(name, n_ent, n_rel, dim,
                        num_bases=num_bases, dropout=dropout).to(device)

    adj_list = (build_adj(train, n_ent, n_rel, device)
                if model.encoder_kind == "rgcn" else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    nq = len(rows)
    best_val, best_test, best_pat = -1.0, None, None
    best_state, best_ep, diverged_ep, last_ep = None, None, None, None  # [guard]
    bad = 0
    start_ep = 1
    prior_elapsed = 0.0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["ep"] + 1
        best_val = ck["best_val"]
        best_test = ck["best_test"]
        best_pat = ck["best_pat"]
        bad = ck["bad"]
        prior_elapsed = ck["elapsed"]
        best_state = ck.get("best_state")              # [guard]
        best_ep = ck.get("best_ep")
        if "rng_torch" in ck:          # restore the exact RNG stream
            torch.set_rng_state(ck["rng_torch"])

    def save_ckpt(ep):
        if ckpt_path is None:
            return
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ep": ep, "best_val": best_val, "best_test": best_test,
                    "best_pat": best_pat, "bad": bad,
                    "best_state": best_state, "best_ep": best_ep,  # [guard]
                    "elapsed": prior_elapsed + (time.time() - t0),
                    # RNG state must be persisted: without it a resumed run
                    # draws a different torch.randperm sequence than an
                    # uninterrupted one and converges to a different model.
                    # Only torch's RNG is persisted: it is the sole
                    # generator consumed inside the training loop
                    # (torch.randperm for batch order, and dropout), and a
                    # ByteTensor stays loadable under torch>=2.6's default
                    # weights_only=True.
                    "rng_torch": torch.get_rng_state()},
                   ckpt_path)

    t0 = time.time()
    is_rgcn = (model.encoder_kind == "rgcn")             # [guard]
    for ep in range(start_ep, epochs + 1):
        model.train()
        perm = torch.randperm(nq, device=device)
        ep_loss = 0.0
        is_rgcn = (model.encoder_kind == "rgcn")
        nb = (nq + bs - 1) // bs
        for bi in range(nb):
            idx = perm[bi * bs:(bi + 1) * bs]
            qb = Q[idx]
            a, r, d = qb[:, 0], qb[:, 1], qb[:, 2]
            # full-graph encoder is recomputed every mini-batch so that
            # the relational encoder receives one optimisation step per
            # mini-batch (parity with the factorisation baselines).
            E = model.encode(adj_list) if is_rgcn else model.encode()
            opt.zero_grad()
            tail_mask = (d == 0)
            logits = torch.empty(len(idx), n_ent, device=device)
            if tail_mask.any():
                logits[tail_mask] = model.score_all_t(
                    E, a[tail_mask], r[tail_mask])
            if (~tail_mask).any():
                logits[~tail_mask] = model.score_all_h(
                    E, r[~tail_mask], a[~tail_mask])
            tgt = torch.zeros_like(logits)
            flat_rows, flat_cols = [], []
            for row, ii in enumerate(idx.tolist()):
                c = qtgt[ii]
                flat_cols.append(c)
                flat_rows.append(torch.full((c.numel(),), row,
                                            dtype=torch.long, device=device))
            fr = torch.cat(flat_rows); fc = torch.cat(flat_cols)
            tgt[fr, fc] = 1.0
            tgt = tgt * (1.0 - label_smooth) + label_smooth / n_ent
            loss = F.binary_cross_entropy_with_logits(logits, tgt)
            if reg > 0:
                loss = loss + reg * model.regularizer(a, r, a)
            if not torch.isfinite(loss):                # [guard]
                diverged_ep = ep        # parameters are no longer usable
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
        last_ep = ep
        if diverged_ep is not None:                     # [guard]
            break

        if ep % eval_every == 0 or ep == epochs:
            # [guard] encode in eval mode: the GNN layers contain dropout,
            # and encoding in train mode evaluated a randomly thinned
            # network (R-GCN test MRR 0.028 instead of 0.041 in a check)
            model.eval()
            with torch.no_grad():
                E = (model.encode(adj_list) if is_rgcn else model.encode())
            vres, _ = evaluate(model, valid, E, t_filt, h_filt,
                               n_ent, device)
            if vres["MRR"] > best_val + 1e-5:
                best_val = vres["MRR"]
                best_test, best_pat = evaluate(
                    model, test, E, t_filt, h_filt, n_ent, device,
                    rel_pattern, id2rel, type_masks=type_masks)
                best_state = {k: v.detach().clone()      # [guard]
                              for k, v in model.state_dict().items()}
                best_ep = ep
                bad = 0
            else:
                bad += 1
            if verbose:
                print(f"  [{name}] ep{ep:3d} loss={ep_loss:.3f} "
                      f"valMRR={vres['MRR']:.4f} best={best_val:.4f}")
            if bad >= patience:
                break
            save_ckpt(ep)            # durable progress at each eval

        if (time_budget is not None
                and (time.time() - t0) > time_budget):
            save_ckpt(ep)
            return {"status": "incomplete", "model": name,
                    "last_ep": ep, "best_val": best_val}

    if best_test is None:                               # [guard]
        # diverged before the first evaluation: score what is left, so that
        # the run is recorded (as near-random) instead of crashing
        model.eval()                                     # [guard]
        with torch.no_grad():
            E = model.encode(adj_list) if is_rgcn else model.encode()
        vres, _ = evaluate(model, valid, E, t_filt, h_filt, n_ent, device)
        best_val = vres["MRR"]
        best_test, best_pat = evaluate(
            model, test, E, t_filt, h_filt, n_ent, device,
            rel_pattern, id2rel, type_masks=type_masks)
    if best_state is not None:
        # hand back the model that produced the reported numbers, not the
        # last one: stage 2 and the warm start then begin exactly there
        model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return {"model": name, "test": best_test, "pattern": best_pat,
            "val_mrr": best_val, "params": int(n_params),
            "best_epoch": best_ep, "last_epoch": last_ep,
            "diverged_epoch": diverged_ep,
            "train_time_s": round(prior_elapsed + time.time() - t0, 1),
            "trained_model": model}
