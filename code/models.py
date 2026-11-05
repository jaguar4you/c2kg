"""
models.py
---------
Link-prediction models evaluated on C2KG.

  * TransE   (Bordes et al., 2013)        translational distance
  * RotatE   (Sun et al., 2019)           rotation in complex space
  * DistMult (Yang et al., 2015)          diagonal bilinear (symmetric)
  * ComplEx  (Trouillon et al., 2016)     Hermitian bilinear
  * RGCN     (Schlichtkrull et al., 2018) residual R-GCN encoder + DistMult
  * CompGCN  (Vashishth et al., 2020)     residual CompGCN encoder + DistMult

Every model exposes encode(), score_all_t(E, h, r) and score_all_h(E, r, t),
which score all candidate tails or heads at once, and regularizer().  The
models are trained with the 1-N (KvsAll) binary cross-entropy of
train_eval.py.  The warm-started encoder (WS-RGCN) is in warmstart_gnn.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Decoder-only factorisation baselines
# ----------------------------------------------------------------------------
class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim=200, margin=9.0):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        self.encoder_kind = "none"

    def encode(self):
        return self.ent.weight

    def score(self, E, h, r, t):
        s = E[h] + self.rel(r) - E[t]
        return self.margin - torch.norm(s, p=2, dim=-1)

    def _dist2_all(self, q, E):
        # ||q - e||^2 = ||q||^2 - 2 q.e^T + ||e||^2  (matmul form, fast)
        qn = (q * q).sum(-1, keepdim=True)           # (B,1)
        en = (E * E).sum(-1)                          # (N,)
        return (qn - 2.0 * q @ E.t() + en).clamp_min(1e-12)

    def score_all_t(self, E, h, r):
        q = E[h] + self.rel(r)                        # (B, d)
        return self.margin - torch.sqrt(self._dist2_all(q, E))

    def score_all_h(self, E, r, t):
        q = E[t] - self.rel(r)
        return self.margin - torch.sqrt(self._dist2_all(q, E))

    def regularizer(self, h, r, t):
        return torch.tensor(0.0, device=self.ent.weight.device)


class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, dim=200):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        self.encoder_kind = "none"

    def encode(self):
        return self.ent.weight

    def score(self, E, h, r, t):
        return torch.sum(E[h] * self.rel(r) * E[t], dim=-1)

    def score_all_t(self, E, h, r):
        return (E[h] * self.rel(r)) @ E.t()

    def score_all_h(self, E, r, t):
        return (self.rel(r) * E[t]) @ E.t()

    def regularizer(self, h, r, t):
        return (E_norm(self.ent, h) + E_norm(self.ent, t) +
                self.rel(r).pow(2).sum(-1).mean())


class ComplEx(nn.Module):
    """Hermitian-product factorisation.  Embedding stored as [real | imag]."""
    def __init__(self, n_ent, n_rel, dim=200):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(n_ent, 2 * dim)
        self.rel = nn.Embedding(n_rel, 2 * dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        self.encoder_kind = "none"

    def encode(self):
        return self.ent.weight

    @staticmethod
    def _ri(x, d):
        return x[..., :d], x[..., d:]

    def score(self, E, h, r, t):
        d = self.dim
        hr, hi = self._ri(E[h], d)
        rr, ri = self._ri(self.rel(r), d)
        tr, ti = self._ri(E[t], d)
        # Re(<h, r, conj(t)>)
        return torch.sum(
            (hr * rr - hi * ri) * tr + (hr * ri + hi * rr) * ti, dim=-1)

    def score_all_t(self, E, h, r):
        d = self.dim
        hr, hi = self._ri(E[h], d)
        rr, ri = self._ri(self.rel(r), d)
        Er, Ei = E[:, :d], E[:, d:]
        a = hr * rr - hi * ri
        b = hr * ri + hi * rr
        return a @ Er.t() + b @ Ei.t()

    def score_all_h(self, E, r, t):
        d = self.dim
        rr, ri = self._ri(self.rel(r), d)
        tr, ti = self._ri(E[t], d)
        Er, Ei = E[:, :d], E[:, d:]
        # score(h) = <hr,(rr*tr+ri*ti)> + <hi,(rr*ti-ri*tr)>
        a = rr * tr + ri * ti
        b = rr * ti - ri * tr
        return a @ Er.t() + b @ Ei.t()

    def regularizer(self, h, r, t):
        return (self.ent(h).pow(2).sum(-1).mean() +
                self.ent(t).pow(2).sum(-1).mean() +
                self.rel(r).pow(2).sum(-1).mean())


class RotatE(nn.Module):
    """Relational rotation in complex space (Sun et al., ICLR 2019).

    Entities are complex vectors stored as [real | imag]; a relation is a
    rotation, i.e. a unit-modulus complex vector parameterised by a phase
    so that |r_k| = 1 by construction.  The score is the negative L2
    distance between the rotated head and the tail,
        phi(h,r,t) = margin - || h o r - t ||_2 ,
    where 'o' is the elementwise complex (Hadamard) product.  For
    one-vs-all scoring the squared distance is expanded into a matmul
    form so that all candidates are scored with two dense products.
    """
    def __init__(self, n_ent, n_rel, dim=200, margin=9.0):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.ent = nn.Embedding(n_ent, 2 * dim)
        # relation phases in (-pi, pi]; modulus fixed to one
        self.rel_phase = nn.Embedding(n_rel, dim)
        nn.init.uniform_(self.ent.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel_phase.weight, -3.14159265, 3.14159265)
        self.encoder_kind = "none"
        # alias so that generic code paths that touch `.rel` still work
        self.rel = self.rel_phase

    def encode(self):
        return self.ent.weight

    def _rot(self, E, h, r):
        """Rotated head embedding h o r, returned as (real, imag)."""
        d = self.dim
        hr, hi = E[h][..., :d], E[h][..., d:]
        phase = self.rel_phase(r)
        rr, ri = torch.cos(phase), torch.sin(phase)
        return hr * rr - hi * ri, hr * ri + hi * rr

    def _inv_rot(self, E, r, t):
        """Inverse-rotated tail t o conj(r), returned as (real, imag)."""
        d = self.dim
        tr, ti = E[t][..., :d], E[t][..., d:]
        phase = self.rel_phase(r)
        rr, ri = torch.cos(phase), torch.sin(phase)
        return tr * rr + ti * ri, ti * rr - tr * ri

    def _dist_all(self, qr, qi, E):
        """margin - ||q - e||_2 for every candidate e, via matmul."""
        d = self.dim
        Er, Ei = E[:, :d], E[:, d:]
        qn = (qr * qr).sum(-1, keepdim=True) + (qi * qi).sum(-1, keepdim=True)
        en = (Er * Er).sum(-1) + (Ei * Ei).sum(-1)
        d2 = (qn - 2.0 * (qr @ Er.t() + qi @ Ei.t()) + en).clamp_min(1e-12)
        return self.margin - torch.sqrt(d2)

    def score(self, E, h, r, t):
        d = self.dim
        qr, qi = self._rot(E, h, r)
        tr, ti = E[t][..., :d], E[t][..., d:]
        s = torch.sqrt(((qr - tr) ** 2 + (qi - ti) ** 2).sum(-1) + 1e-12)
        return self.margin - s

    def score_all_t(self, E, h, r):
        qr, qi = self._rot(E, h, r)
        return self._dist_all(qr, qi, E)

    def score_all_h(self, E, r, t):
        qr, qi = self._inv_rot(E, r, t)
        return self._dist_all(qr, qi, E)

    def regularizer(self, h, r, t):
        # modulus of relations is fixed to 1, so only entities are penalised
        return (self.ent(h).pow(2).sum(-1).mean() +
                self.ent(t).pow(2).sum(-1).mean())


def E_norm(emb, idx):
    return emb(idx).pow(2).sum(-1).mean()


# ----------------------------------------------------------------------------
# Relational graph encoder
# ----------------------------------------------------------------------------
class RGCNLayer(nn.Module):
    """
    One R-GCN layer.  Supports vanilla per-relation weights, basis
    decomposition, and (optional) attention-weighted aggregation.
    Adjacency is supplied as a list of (rel_id, norm_sparse_adj) pairs that
    already includes inverse relations and self-loop handled by W0.
    """
    def __init__(self, in_dim, out_dim, n_rel, num_bases=0,
                 attention=False, dropout=0.0, zero_init=False):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.n_rel = n_rel
        self.num_bases = num_bases
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.W0 = nn.Parameter(torch.empty(in_dim, out_dim))

        if num_bases and num_bases > 0:
            self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
            self.coeff = nn.Parameter(torch.empty(n_rel, num_bases))
        else:
            self.Wr = nn.Parameter(torch.empty(n_rel, in_dim, out_dim))

        if attention:
            self.att = nn.Parameter(torch.empty(n_rel, 2 * out_dim))

        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init=False):
        """Xavier initialisation, or all zeros when ``zero_init`` is set."""
        if zero_init:
            nn.init.zeros_(self.W0)
        else:
            nn.init.xavier_uniform_(self.W0)
        if self.num_bases and self.num_bases > 0:
            if zero_init:
                nn.init.zeros_(self.bases); nn.init.zeros_(self.coeff)
            else:
                nn.init.xavier_uniform_(self.bases)
                nn.init.xavier_uniform_(self.coeff)
        else:
            (nn.init.zeros_ if zero_init
             else nn.init.xavier_uniform_)(self.Wr)
        if self.attention:
            (nn.init.zeros_ if zero_init
             else nn.init.xavier_uniform_)(self.att)

    def relation_weight(self, r):
        if self.num_bases and self.num_bases > 0:
            return torch.einsum("b,bio->io", self.coeff[r], self.bases)
        return self.Wr[r]

    def forward(self, H, adj_list):
        out = H @ self.W0                                    # self loop
        for r, A in adj_list:                                # A: sparse (N,N)
            Wr = self.relation_weight(r)
            msg = torch.sparse.mm(A, H @ Wr)
            if self.attention:
                # gating by relation-specific attention on aggregated msg
                agg = H @ Wr
                cat = torch.cat([torch.sparse.mm(A, agg), out], dim=-1)
                g = torch.sigmoid(cat @ self.att[r])
                msg = msg * g.unsqueeze(-1)
            out = out + msg
        return self.dropout(F.relu(out))


class RGCNEncoder(nn.Module):
    """Two-layer residual relational encoder over learned input embeddings:
    H <- H + layer(H) for each of the two layers."""
    def __init__(self, n_ent, n_rel, in_dim, hid_dim, out_dim,
                 num_bases=0, attention=False, dropout=0.2,
                 learn_input=True, residual=True, zero_init=False):
        super().__init__()
        assert in_dim == hid_dim == out_dim, \
            "residual encoder requires equal in/hid/out dims"
        self.learn_input = learn_input
        self.residual = residual
        self.input = nn.Parameter(torch.empty(n_ent, in_dim))
        nn.init.xavier_uniform_(self.input)
        self.l1 = RGCNLayer(in_dim, hid_dim, 2 * n_rel, num_bases,
                            attention, dropout, zero_init=zero_init)
        self.l2 = RGCNLayer(hid_dim, out_dim, 2 * n_rel, num_bases,
                            attention, dropout, zero_init=zero_init)

    def forward(self, adj_list):
        h = self.input
        h = h + self.l1(h, adj_list) if self.residual else self.l1(h, adj_list)
        h = h + self.l2(h, adj_list) if self.residual else self.l2(h, adj_list)
        return h


# ----------------------------------------------------------------------------
# Encoder-decoder models
# ----------------------------------------------------------------------------
class RGCN(nn.Module):
    """R-GCN encoder + DistMult decoder (Schlichtkrull et al., 2018)."""
    def __init__(self, n_ent, n_rel, dim=200, num_bases=10, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.enc = RGCNEncoder(n_ent, n_rel, dim, dim, dim,
                               num_bases=num_bases, dropout=dropout)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.rel.weight)
        self.encoder_kind = "rgcn"

    def encode(self, adj_list):
        return self.enc(adj_list)

    def score(self, E, h, r, t):
        return torch.sum(E[h] * self.rel(r) * E[t], dim=-1)

    def score_all_t(self, E, h, r):
        return (E[h] * self.rel(r)) @ E.t()

    def score_all_h(self, E, r, t):
        return (self.rel(r) * E[t]) @ E.t()

    def regularizer(self, h, r, t):
        return self.rel(r).pow(2).sum(-1).mean()


class CompGCNLayer(nn.Module):
    """One CompGCN layer (Vashishth et al., ICLR 2020).

    Unlike R-GCN, which learns one weight matrix per relation, CompGCN
    composes the neighbour entity with the *relation embedding* using a
    non-parametric operator phi (here the multiplicative and subtraction
    operators are supported) and then applies a single direction-specific
    weight (in / out / self).  This keeps the parameter count independent
    of |R| and lets the relation embeddings themselves be updated by a
    learned projection W_rel.
    """
    def __init__(self, in_dim, out_dim, comp="mult", dropout=0.0):
        super().__init__()
        self.comp = comp
        self.dropout = nn.Dropout(dropout)
        self.W_in = nn.Parameter(torch.empty(in_dim, out_dim))
        self.W_out = nn.Parameter(torch.empty(in_dim, out_dim))
        self.W_self = nn.Parameter(torch.empty(in_dim, out_dim))
        self.W_rel = nn.Parameter(torch.empty(in_dim, out_dim))
        for p in (self.W_in, self.W_out, self.W_self, self.W_rel):
            nn.init.xavier_uniform_(p)

    def _compose(self, h_u, r_vec):
        if self.comp == "sub":
            return h_u - r_vec
        return h_u * r_vec              # default: multiplicative

    def forward(self, H, R, adj_list, n_rel):
        """H: (n_ent, in_dim) entity states; R: (n_rel*2, in_dim) relation
        states; adj_list: list of (rel_id, sparse_adj) with inverse
        relations already appended as ids >= n_rel."""
        out = H @ self.W_self
        for r, A in adj_list:
            # compose every source state with this relation's embedding
            msg = self._compose(H, R[r].unsqueeze(0))
            W = self.W_out if r < n_rel else self.W_in
            out = out + torch.sparse.mm(A, msg) @ W
        R_next = R @ self.W_rel
        return self.dropout(F.relu(out)), R_next


class CompGCN(nn.Module):
    """CompGCN encoder with a DistMult decoder (the configuration reported
    in the original paper for link prediction).

    Residual layers, H <- H + layer(H), as in the R-GCN encoder of this
    file.  Without them the two-layer encoder at d = 64 does not fit the
    training graph: on C2KG-N its symmetric-pattern tail MRR stayed at
    0.02-0.03 although about 80 % of those test triples have their mirror
    in training (a DistMult decoder scores a mirror identically), and
    changing ReLU to tanh or adding batch normalisation and 1/3 averaging
    as in the official code did not help."""
    def __init__(self, n_ent, n_rel, dim=200, comp="mult", dropout=0.2,
                 n_layers=2):
        super().__init__()
        self.dim = dim
        self.n_rel = n_rel
        self.ent0 = nn.Parameter(torch.empty(n_ent, dim))
        self.rel0 = nn.Parameter(torch.empty(2 * n_rel, dim))
        nn.init.xavier_uniform_(self.ent0)
        nn.init.xavier_uniform_(self.rel0)
        self.layers = nn.ModuleList(
            [CompGCNLayer(dim, dim, comp=comp, dropout=dropout)
             for _ in range(n_layers)])
        self.encoder_kind = "rgcn"      # needs the adjacency at encode time
        self._R = None                  # relation states after encoding

    def encode(self, adj_list):
        H, R = self.ent0, self.rel0
        for layer in self.layers:
            H_new, R = layer(H, R, adj_list, self.n_rel)
            H = H + H_new                               # residual
        self._R = R
        return H

    def _rel(self, r):
        R = self._R if self._R is not None else self.rel0
        return R[r]

    def score(self, E, h, r, t):
        return torch.sum(E[h] * self._rel(r) * E[t], dim=-1)

    def score_all_t(self, E, h, r):
        return (E[h] * self._rel(r)) @ E.t()

    def score_all_h(self, E, r, t):
        return (self._rel(r) * E[t]) @ E.t()

    def regularizer(self, h, r, t):
        return self._rel(r).pow(2).sum(-1).mean()


def build_model(name, n_ent, n_rel, dim=200, num_bases=10, dropout=0.2):
    name = name.lower()
    if name == "transe":
        return TransE(n_ent, n_rel, dim)
    if name == "distmult":
        return DistMult(n_ent, n_rel, dim)
    if name == "complex":
        return ComplEx(n_ent, n_rel, dim)
    if name == "rotate":
        return RotatE(n_ent, n_rel, dim)
    if name == "compgcn":
        return CompGCN(n_ent, n_rel, dim, dropout=dropout)
    if name == "rgcn":
        return RGCN(n_ent, n_rel, dim, num_bases=num_bases, dropout=dropout)
    raise ValueError(name)
