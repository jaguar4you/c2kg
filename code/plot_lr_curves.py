"""
plot_lr_curves.py
-----------------
Figure of the validation MRR across the learning rates of the stage-1
search on C2KG-L (written by make_tables.py to ../results/paper/lr_curves.json).
Requires matplotlib.

    python plot_lr_curves.py ../results/paper/lr_curves.json ../results/paper/fig_lr_curves
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

C = json.load(open(sys.argv[1]))
sel = {"TransE": 0.3, "RotatE": 0.3, "DistMult": 0.3, "ComplEx": 0.3,
       "R-GCN": 0.03, "CompGCN": 0.03}
order = ["RotatE", "ComplEx", "DistMult", "TransE", "R-GCN", "CompGCN"]
col = dict(zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                       "#e87ba4", "#008300"]))
mk = dict(zip(order, ["o", "s", "^", "D", "v", "P"]))
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.3, 3.6))
ax.axvline(1e-2, color="#8a8a85", lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(1.12e-2, 0.188, "shared rate $10^{-2}$", color="#55554f", fontsize=8,
        va="top")
for m in order:
    xs = [p[0] for p in C[m]]
    ys = [p[1] for p in C[m]]
    ax.plot(xs, ys, color=col[m], lw=1.6, marker=mk[m], ms=4.5, zorder=3,
            label=m)
    i = xs.index(sel[m])
    ax.plot([xs[i]], [ys[i]], marker=mk[m], ms=8.5, color=col[m],
            mec="white", mew=1.2, zorder=4)
# direct labels where there is room
where = {"RotatE": (sel["RotatE"], (7, 0)), "ComplEx": (sel["ComplEx"], (7, 3)),
         "DistMult": (1.0, (6, -3)), "TransE": (10.0, (-14, 7)),
         "R-GCN": (sel["R-GCN"], (-12, 7)), "CompGCN": (0.1, (5, -11))}
for m in order:
    xs = [p[0] for p in C[m]]; ys = [p[1] for p in C[m]]
    x0, o = where[m]
    i = xs.index(x0)
    ax.annotate(m, (xs[i], ys[i]), xytext=o, textcoords="offset points",
                fontsize=8, color="#2b2b28")
ax.set_xscale("log")
ax.set_xlim(6e-4, 1.6e1)
ax.set_ylim(0, 0.19)
ax.set_xlabel("Learning rate (stage 1)")
ax.set_ylabel("Validation MRR (seed 0)")
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.2), handlelength=1.8, columnspacing=1.2)
fig.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(sys.argv[2] + "." + ext, dpi=200)
