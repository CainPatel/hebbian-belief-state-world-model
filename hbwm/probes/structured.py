"""Study 2 structured probe families (spec sections 4.1 to 4.5, 5.1, 5.2).

Study 1's `hbwm/probes/probe.py` is untouched: this module holds the Study 2 siblings. Every family is
an nn.Module mapping a *flat* feature row [B, F] to class logits [B, C] and standardizing internally
with the same per-feature mean/std buffers `LinearProbe` uses, so the streamed evaluation path is
shared.
"""

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn

from hbwm.bdh.core import HBWMCore
from hbwm.bdh.upstream.bdh import Attention
from hbwm.probes.probe import STATS_MAX_ELEMS, feature_stats  # noqa: F401

STUDY2_RANKS = (1, 4, 16)  # spec 4.2, selected on probe_val
STUDY2_RESTARTS = 3  # spec 4.7 item 2, factorized families only
MLP_HIDDEN = 512  # spec 4.5
RANDPROJ_DIM = 4096  # spec 4.5
RANDPROJ_DENSITY = 64  # nonzeros per output dimension; see Task 3 spec amendment

FACTORIZED = ("query_rank_r", "shared_query_rank_r", "derot_query_rank_r")


@dataclasses.dataclass(frozen=True)
class StateShape:
    """One model's state seen as `n_heads` matrices of shape [rows, cols] (spec 5.1)."""

    n_heads: int
    rows: int
    cols: int
    rotary: bool = False

    @property
    def n_features(self) -> int:
        return self.n_heads * self.rows * self.cols

    @property
    def saturation_rank(self) -> int:
        """`query_rank_r` is expressivity-equivalent to `flat_linear` at or above this r (spec 5.2)."""
        return min(self.rows, self.cols)

    @property
    def shared_saturation_rank(self) -> int:
        """`shared_query_rank_r` needs the shared basis to span the row space, so r >= rows (spec 5.2)."""
        return self.rows

    def rank_fraction(self, r: int) -> float:
        """Effective rank fraction r / min(P, Q), clipped at 1.0 (spec 5.2 reporting requirement)."""
        return min(1.0, r / self.saturation_rank)


def state_shape(model, feature: str = "sigma_full") -> StateShape:
    if isinstance(model, HBWMCore):
        if feature != "sigma_full":
            raise ValueError(f"no matrix reshape for BDH feature {feature!r}")
        c = model.hcfg
        return StateShape(n_heads=c.n_head, rows=c.n_neurons, cols=c.n_embd, rotary=True)
    cfg = model.cfg
    if hasattr(cfg, "hidden"):  # LSTM: state_vector concatenates, per layer, h_i then c_i
        shape = StateShape(n_heads=1, rows=2 * cfg.n_layer, cols=cfg.hidden)
    else:  # RWKV: per block aa, bb, pp, x_prev_timemix, x_prev_channelmix
        shape = StateShape(n_heads=1, rows=5 * cfg.n_layer, cols=cfg.n_embd)
    assert shape.n_features == model.state_dim, (shape, model.state_dim)
    return shape


def param_count(family: str, shape: StateShape, n_classes: int, rank: int | None = None,
                n_input: int | None = None) -> int:
    """Learnable parameters excluding the per-class bias (spec Appendix B).

    `n_input` is required for the mlp families, whose input width is set outside the module: the full
    state for `mlp_state` (524,288 on BDH), 8,192 for `mlp_rownorm`, 4,096 for `mlp_randproj`. The
    results table reports this for every arm so the degeneracy call of spec 7 can be audited.

    This is the preregistered WEIGHT count: it excludes both the per-class bias every family shares
    and, for the mlp families, the hidden-layer bias `MLPProbe` also carries.
    """
    C, nh, P, Q = n_classes, shape.n_heads, shape.rows, shape.cols
    if family.startswith("mlp_"):
        if n_input is None:
            raise ValueError(f"{family} needs n_input")
        return n_input * MLP_HIDDEN + MLP_HIDDEN * C
    if family in ("flat_linear", "derot_flat_linear"):
        return C * nh * P * Q
    if family in ("query_rank_r", "derot_query_rank_r"):
        if rank is None:
            raise ValueError(f"{family} needs a rank")
        return C * nh * rank * (P + Q)
    if family == "shared_query_rank_r":
        if rank is None:
            raise ValueError(f"{family} needs a rank")
        return nh * rank * P + C * nh * rank * Q
    raise ValueError(f"unknown family {family!r}")


def _buf(v, n: int, fill: float) -> torch.Tensor:
    if v is None:
        return torch.full((n,), fill, dtype=torch.float32)
    return torch.as_tensor(v, dtype=torch.float32).detach().reshape(n).clone()


class StructuredProbe(nn.Module):
    """Flat row [B, F] -> logits [B, C], standardizing internally like `LinearProbe`.

    `set_positions` is the hook the derotated families need: the streamed evaluation path calls
    `probe(xb)` with no extra arguments, so the runner stashes the batch's absolute token positions on
    the module first. Every other family ignores it.
    """

    def __init__(self, n_features: int, n_classes: int, mean=None, std=None):
        super().__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.register_buffer("mean", _buf(mean, n_features, 0.0))
        self.register_buffer("std", _buf(std, n_features, 1.0))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self._positions = None

    def set_positions(self, t) -> None:
        self._positions = None if t is None else torch.as_tensor(t, dtype=torch.float32)

    def standardize(self, x):
        return (x - self.mean) / self.std

    def readout_parameters(self):
        """Parameters the L2 penalty applies to (spec 4.7 item 1): every weight, no biases.

        Excludes any parameter whose name ends in "bias", not just the one literally named "bias":
        this also excludes MLPProbe's hidden-layer bias (named "net.0.bias"), so the penalty is
        exactly ||W||^2 for the flat and MLP families and ||q||^2 + ||v||^2 for the factorized ones.
        """
        return [p for n, p in self.named_parameters() if not n.endswith("bias")]


class FlatLinearProbe(StructuredProbe):
    """Spec 4.1. Study 1's sigma_full probe, refit under Study 2 conditions as the control."""

    def __init__(self, shape: StateShape, n_classes: int, mean=None, std=None, gen=None):
        super().__init__(shape.n_features, n_classes, mean, std)
        self.shape = shape
        w = torch.empty(n_classes, shape.n_features)
        nn.init.normal_(w, std=shape.n_features**-0.5, generator=gen)
        self.weight = nn.Parameter(w)

    def forward(self, x, t=None):
        return self.standardize(x) @ self.weight.T + self.bias


class QueryRankProbe(StructuredProbe):
    """Spec 4.2 (per-class queries) and 4.3 (`shared_query=True`)."""

    def __init__(self, shape: StateShape, n_classes: int, rank: int, mean=None, std=None,
                 shared_query: bool = False, gen=None):
        super().__init__(shape.n_features, n_classes, mean, std)
        self.shape, self.rank, self.shared_query = shape, rank, shared_query
        nh, P, Q = shape.n_heads, shape.rows, shape.cols
        q = torch.empty((nh, rank, P) if shared_query else (n_classes, nh, rank, P))
        v = torch.empty(n_classes, nh, rank, Q)
        # spec 4.7 item 2: q ~ N(0, 1/N), v ~ N(0, 1/D), i.e. std P^-0.5 and Q^-0.5.
        nn.init.normal_(q, std=P**-0.5, generator=gen)
        nn.init.normal_(v, std=Q**-0.5, generator=gen)
        self.q, self.v = nn.Parameter(q), nn.Parameter(v)

    def flat_weight(self):
        """Implied per-class weight W[c, h, p, q], rank <= self.rank per head."""
        if self.shared_query:
            return torch.einsum("hjp,chjq->chpq", self.q, self.v)
        return torch.einsum("chjp,chjq->chpq", self.q, self.v)

    def forward(self, x, t=None):
        s = self.shape
        z = self.standardize(x).view(-1, s.n_heads, s.rows, s.cols)
        return torch.einsum("bhpq,chpq->bc", z, self.flat_weight()) + self.bias


class MLPProbe(StructuredProbe):
    """Spec 4.5 capacity control: n_features -> hidden -> n_classes, ReLU.

    One class for all three family-5 members, because the reduction happens outside the module and
    only the input width differs: `mlp_state` (the full state vector: 524,288 on BDH, 1,400 on the
    LSTM, 3,520 on RWKV, and the matched arm H6 uses), `mlp_rownorm` (sigma_rownorm, 8,192 dims,
    BDH only) and `mlp_randproj` (the sparse projection of flat sigma, 4,096 dims, BDH only).
    """

    def __init__(self, n_features: int, n_classes: int, hidden: int = MLP_HIDDEN, mean=None, std=None,
                 gen=None):
        super().__init__(n_features, n_classes, mean, std)
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes, bias=False))
        for m in (self.net[0], self.net[2]):
            nn.init.normal_(m.weight, std=m.in_features**-0.5, generator=gen)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, x, t=None):
        return self.net(self.standardize(x)) + self.bias


def sparse_randproj(n_in: int, n_out: int, density: int = RANDPROJ_DENSITY, seed: int = 0):
    """Very sparse random projection: `density` signed nonzeros per output dimension (spec 4.5)."""
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(n_in, size=density, replace=False) for _ in range(n_out)]).astype(np.int64)
    sign = rng.choice([-1.0, 1.0], size=(n_out, density)).astype(np.float32)
    return idx, sign


def apply_randproj(x, idx, sign) -> np.ndarray:
    """x: [B, n_in] -> [B, n_out] float32. Gathers `density` columns per output dimension."""
    x = np.asarray(x, dtype=np.float32)
    return np.einsum("bod,od->bo", x[:, idx], sign, optimize=True).astype(np.float32)


def derotate(sigma, t, freqs):
    """Apply R(-t) along the row (neuron) axis of sigma [..., nh, P, Q] (spec 4.4, Appendix A).

    `Attention.rope` rotates interleaved pairs of the LAST axis, so transpose the row axis into place,
    rotate with phases -t * freqs, and transpose back. `t` is the absolute TOKEN position of the
    feature timestep, i.e. `tokenizer.obs_positions(L)[t_env]`, not the environment step index.
    """
    t = torch.as_tensor(t, dtype=torch.float32, device=sigma.device).reshape(-1)
    phases = (-t).view(-1, 1, 1, 1) * freqs.to(sigma.device).view(1, 1, 1, -1)
    return Attention.rope(phases, sigma.transpose(-1, -2)).transpose(-1, -2)


class DerotProbe(StructuredProbe):
    """Spec 4.4: derotate, then delegate to any inner family.

    On a state with no rotary phase there is nothing to undo, so the matched baseline definition is the
    inner probe itself (spec 5.1); `build_family` returns the bare inner probe in that case and never
    constructs this wrapper.
    """

    def __init__(self, inner: StructuredProbe, shape: StateShape, freqs, mean=None, std=None):
        super().__init__(shape.n_features, inner.n_classes, mean, std)
        self.inner, self.shape = inner, shape
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32).reshape(shape.rows))
        del self.bias  # the inner probe owns the bias

    def set_positions(self, t) -> None:
        super().set_positions(t)
        self.inner.set_positions(None)

    def readout_parameters(self):
        return self.inner.readout_parameters()

    def forward(self, x, t=None):
        pos = self._positions if t is None else t
        if pos is None:
            raise ValueError("DerotProbe needs absolute token positions: pass t or call set_positions")
        s = self.shape
        # Derotate the RAW row FIRST, then standardize in the derotated frame (spec 4.4). The two do
        # not commute: standardization scales each (h, n, d) separately while the rotation mixes the
        # pairs (2j, 2j+1) at fixed d, so the other order rotates a distorted space instead of undoing
        # RoPE. The inner probe is built with identity mean/std, and because the order lives here both
        # fitting and the streamed passes get exactly the same input distribution.
        z = x.view(-1, s.n_heads, s.rows, s.cols)
        return self.inner(self.standardize(derotate(z, pos, self.freqs).reshape(x.shape[0], -1)))


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    name: str
    rank: int | None = None
    input_kind: str = "flat"  # flat | rownorm | randproj | state
    n_restarts: int = 1


def spec_label(spec: FamilySpec) -> str:
    return spec.name if spec.rank is None else spec.name.replace("_rank_r", f"_rank_{spec.rank}")


def family_specs(shape: StateShape) -> list[FamilySpec]:
    """The preregistered Study 2 family set for one state (spec 4.1 to 4.5, 5.1).

    Family 6 (`synapse_atlas`) is exploratory and is not in this list. On a non-rotary state the two
    derot families ARE their undecorated counterparts, so they are not refitted; the aggregation records
    them as aliases (spec 5.1).
    """
    out = [FamilySpec("flat_linear")]
    out += [FamilySpec("query_rank_r", r, n_restarts=STUDY2_RESTARTS) for r in STUDY2_RANKS]
    out += [FamilySpec("shared_query_rank_r", r, n_restarts=STUDY2_RESTARTS) for r in STUDY2_RANKS]
    if shape.rotary:
        out.append(FamilySpec("derot_flat_linear"))
        out += [FamilySpec("derot_query_rank_r", r, n_restarts=STUDY2_RESTARTS) for r in STUDY2_RANKS]
    out.append(FamilySpec("mlp_state", input_kind="state"))  # family 5's matched arm, every state
    if shape.rotary:  # the two reductions have no baseline counterpart (spec 4.5)
        out += [FamilySpec("mlp_rownorm", input_kind="rownorm"),
                FamilySpec("mlp_randproj", input_kind="randproj")]
    return out


def build_family(spec: FamilySpec, shape: StateShape, n_classes: int, n_input: int, mean=None,
                 std=None, freqs=None, gen=None) -> StructuredProbe:
    name = spec.name
    if name.startswith("mlp_"):
        return MLPProbe(n_input, n_classes, MLP_HIDDEN, mean, std, gen)
    if name.startswith("derot_"):
        plain = FamilySpec(name[len("derot_"):], spec.rank)
        if not shape.rotary:  # spec 5.1: nothing to undo, so the matched arm IS the plain family
            return build_family(plain, shape, n_classes, n_input, mean, std, gen=gen)
        if freqs is None:
            raise ValueError(f"{spec_label(spec)} needs the model's rope frequencies")
        # The wrapper owns standardization (Task 4): the inner probe gets the identity transform.
        inner = build_family(plain, shape, n_classes, n_input, gen=gen)
        return DerotProbe(inner, shape, freqs, mean, std)
    if name == "flat_linear":
        return FlatLinearProbe(shape, n_classes, mean, std, gen)
    if name in ("query_rank_r", "shared_query_rank_r"):
        return QueryRankProbe(shape, n_classes, spec.rank, mean, std,
                              shared_query=(name == "shared_query_rank_r"), gen=gen)
    raise ValueError(f"unknown family {name!r}")
