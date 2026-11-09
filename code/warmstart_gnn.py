"""
warmstart_gnn.py
----------------
WS-RGCN: a residual R-GCN encoder warm-started from a trained RotatE.
(The runner and the results files call it ``re_rgcn``.)

Model
-----
Entity inputs of width 2d are copied from the stage-1 RotatE and
fine-tuned; relation phases are copied from the same RotatE; the decoder is
the RotatE distance with the same margin (9 by default).  Between input
and decoder sit two residual relational blocks over the 2|R| directed
relations,

    H  <-  H + Dropout(ReLU(H W0 + sum_r A_r H W_r)) U ,      U = 0 at init,

with W0, W_r at their ordinary (Xavier) initialisation.  Hence

  * identity start:  U = 0  =>  H_out = H_in exactly, so at initialisation
    the model reproduces the stage-1 RotatE scores (the runner checks this
    and records the largest deviation as ``identity_err``);
  * no blocked gradient:  dL/dU = Z^T dL/dH_out with Z = Dropout(ReLU(...))
    non-zero, so U moves on the first step, and W0, W_r receive gradient
    from the second step on.

The only difference from the stage-2 RotatE run (the continuation control)
is the pair of relational blocks; ``gnn_change`` measures how far the
trained blocks move the embeddings away from their input.
"""

import torch
import torch.nn as nn

from models import RGCNLayer


class ResidualRGCNBlock(nn.Module):
    """H -> H + Dropout(ReLU(H W0 + sum_r A_r H W_r)) U, with U = 0 at init."""

    def __init__(self, dim, n_rel_directed, num_bases=0, dropout=0.2):
        super().__init__()
        # ordinary (non-zero) initialisation inside the nonlinearity
        self.conv = RGCNLayer(dim, dim, n_rel_directed, num_bases,
                              attention=False, dropout=dropout,
                              zero_init=False)
        # the only zero-initialised tensor: the output projection
        self.U = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, H, adj_list):
        return H + self.conv(H, adj_list) @ self.U


class WarmStartRGCN(nn.Module):
    """Trained RotatE -> residual R-GCN -> RotatE decoder."""

    def __init__(self, n_ent, n_rel, dim=64, num_bases=0, dropout=0.2,
                 n_layers=2, margin=9.0):
        super().__init__()
        self.dim, self.margin = dim, margin
        self.input = nn.Parameter(torch.empty(n_ent, 2 * dim))
        nn.init.xavier_uniform_(self.input)
        self.blocks = nn.ModuleList(
            [ResidualRGCNBlock(2 * dim, 2 * n_rel, num_bases, dropout)
             for _ in range(n_layers)])
        self.rel = nn.Embedding(n_rel, dim)            # phases
        nn.init.uniform_(self.rel.weight, -3.14159265, 3.14159265)
        self.encoder_kind = "rgcn"

    # -- warm start ----------------------------------------------------------
    def init_from(self, pre):
        """Copy entity embeddings and relation phases from a trained
        RotatE.  The blocks are already the identity (U = 0)."""
        with torch.no_grad():
            self.input.copy_(pre.ent.weight.detach())
            self.rel.weight.copy_(pre.rel_phase.weight.detach())

    def encode(self, adj_list):
        H = self.input
        for blk in self.blocks:
            H = blk(H, adj_list)
        return H

    # -- RotatE decoder ------------------------------------------------------
    def _split(self, x):
        return x[..., :self.dim], x[..., self.dim:]

    def _rot(self, E, h, r):                  # h o r
        hr, hi = self._split(E[h])
        c, s = torch.cos(self.rel(r)), torch.sin(self.rel(r))
        return hr * c - hi * s, hr * s + hi * c

    def _inv_rot(self, E, r, t):              # t o conj(r)
        tr, ti = self._split(E[t])
        c, s = torch.cos(self.rel(r)), torch.sin(self.rel(r))
        return tr * c + ti * s, ti * c - tr * s

    def _dist_all(self, qr, qi, E):
        Er, Ei = self._split(E)
        qn = (qr * qr).sum(-1, keepdim=True) + (qi * qi).sum(-1, keepdim=True)
        en = (Er * Er).sum(-1) + (Ei * Ei).sum(-1)
        d2 = (qn - 2.0 * (qr @ Er.t() + qi @ Ei.t()) + en).clamp_min(1e-12)
        return self.margin - torch.sqrt(d2)

    def score(self, E, h, r, t):
        qr, qi = self._rot(E, h, r)
        tr, ti = self._split(E[t])
        return self.margin - torch.sqrt(
            ((qr - tr) ** 2 + (qi - ti) ** 2).sum(-1) + 1e-12)

    def score_all_t(self, E, h, r):
        return self._dist_all(*self._rot(E, h, r), E)

    def score_all_h(self, E, r, t):
        return self._dist_all(*self._inv_rot(E, r, t), E)

    def regularizer(self, h, r, t):
        # same terms as RotatE's own regulariser: entity norms only
        return (self.input[h].pow(2).sum(-1).mean() +
                self.input[t].pow(2).sum(-1).mean())

    # -- diagnostics ---------------------------------------------------------
    @torch.no_grad()
    def gnn_change(self, adj_list):
        """||encode - input||_F / ||input||_F in eval mode (no dropout).
        0 means the message passing contributes nothing."""
        was = self.training
        self.eval()
        d = (self.encode(adj_list) - self.input).norm() / self.input.norm()
        self.train(was)
        return float(d)


# name used by the runner and in the results files
WARMSTART_MODELS = {"re_rgcn": "rotate"}


def build_warmstart(name, n_ent, n_rel, dim=64, num_bases=0, dropout=0.2):
    if name not in WARMSTART_MODELS:
        raise ValueError(f"unknown warm-start model {name!r}")
    import models
    return WarmStartRGCN(n_ent, n_rel, dim, num_bases=num_bases,
                         dropout=dropout, n_layers=2, margin=models.MARGIN)
