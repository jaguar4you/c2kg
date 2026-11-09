"""
test_metrics.py
---------------
Checks of the evaluation metrics, runnable without a GPU:

    python test_metrics.py

1. The exact expectations of train_eval.py (reciprocal rank, Hits@k, rank
   under uniformly random tie-breaking) agree with a Monte Carlo simulation
   of random tie-breaking on random score rows with many ties.
2. Without ties they equal the ordinary rank-based definitions.
3. The training-free probes of path_probe.py use the same definitions.
4. A fully tied row gives the random-ranking value H(n)/n, not 2/(n+1).
"""

import random

import numpy as np
import torch

import train_eval as TE
import path_probe as PP


def mc_expect(row, gold_idx, k_list=(1, 3, 10), draws=200000, seed=0):
    """Random tie-breaking by sampling a random secondary key."""
    rng = np.random.default_rng(seed)
    row = np.asarray(row, dtype=np.float64)
    g = int((row > row[gold_idx]).sum())
    tied = np.flatnonzero(row == row[gold_idx])
    m = len(tied)
    # position of the gold among the tied candidates, uniformly at random
    pos = rng.integers(0, m, size=draws)                # 0 .. m-1
    rank = g + 1 + pos
    out = {"rr": float(np.mean(1.0 / rank)), "rank": float(np.mean(rank))}
    for k in k_list:
        out[f"h{k}"] = float(np.mean(rank <= k))
    return out


def check_exact_vs_montecarlo():
    rng = random.Random(1)
    worst = 0.0
    for trial in range(300):
        n = rng.choice([5, 12, 40, 200])
        levels = rng.randint(1, 6)                      # few distinct values
        row = [float(rng.randint(0, levels)) for _ in range(n)]
        gold = rng.randrange(n)
        H = TE._harmonic(n, "cpu")
        g, m = TE._tie_counts(torch.tensor(np.asarray(row)[None, :]), torch.tensor([row[gold]]))
        ex = {k: float(v[0]) for k, v in TE._expected(g, m, H).items()}
        mc = mc_expect(row, gold, seed=trial)
        for key in ("rr", "rank", "h1", "h3", "h10"):
            tol = 0.01 * max(1.0, abs(ex[key])) if key == "rank" else 0.004
            err = abs(ex[key] - mc[key])
            worst = max(worst, err if key != "rank" else err / max(1, ex[key]))
            assert err <= tol, (trial, key, ex[key], mc[key], row, gold)
    return worst


def check_no_ties_equal_ordinary():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(2, 300))
        row = rng.permutation(n).astype(np.float64)    # all distinct
        gold = int(rng.integers(0, n))
        rank = int((row > row[gold]).sum()) + 1
        H = TE._harmonic(n, "cpu")
        g, m = TE._tie_counts(torch.tensor(np.asarray(row)[None, :]), torch.tensor([row[gold]]))
        ex = {k: float(v[0]) for k, v in TE._expected(g, m, H).items()}
        assert int(m[0]) == 1
        assert abs(ex["rr"] - 1.0 / rank) < 1e-12
        assert ex["rank"] == rank
        for k in (1, 3, 10):
            assert ex[f"h{k}"] == float(rank <= k)


def check_probe_definition():
    rng = random.Random(3)
    for _ in range(500):
        g = rng.randint(0, 50)
        m = rng.randint(1, 50)
        H = TE._harmonic(g + m, "cpu")
        ex = TE._expected(torch.tensor([g]), torch.tensor([m]), H)
        rr, h1 = PP._expected(g, m, 1)
        _, h10 = PP._expected(g, m, 10)
        assert abs(rr - float(ex["rr"][0])) < 1e-12
        assert abs(h1 - float(ex["h1"][0])) < 1e-12
        assert abs(h10 - float(ex["h10"][0])) < 1e-12


def check_all_tied():
    n = 1923
    H = TE._harmonic(n, "cpu")
    g, m = TE._tie_counts(torch.zeros(1, n), torch.zeros(1))
    rr = float(TE._expected(g, m, H)["rr"][0])
    want = sum(1.0 / k for k in range(1, n + 1)) / n
    assert abs(rr - want) < 1e-12, (rr, want)
    return rr


if __name__ == "__main__":
    w = check_exact_vs_montecarlo()
    check_no_ties_equal_ordinary()
    check_probe_definition()
    rr = check_all_tied()
    print(f"metrics ok: exact = Monte Carlo (300 tied rows, largest "
          f"deviation {w:.4f}); no ties = ordinary ranks; probes = trainer; "
          f"all-tied row of 1923 = {rr:.6f}")
