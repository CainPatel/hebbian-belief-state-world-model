"""Study 2 structured probe families (spec sections 4.1 to 4.5, 5.1, 5.2).

Study 1's `hbwm/probes/probe.py` is untouched: this module holds the Study 2 siblings. Every family is
an nn.Module mapping a *flat* feature row [B, F] to class logits [B, C] and standardizing internally
with the same per-feature mean/std buffers `LinearProbe` uses, so the streamed evaluation path is
shared.
"""

import dataclasses

import numpy as np  # noqa: F401
import torch  # noqa: F401
import torch.nn.functional as F  # noqa: F401
from torch import nn  # noqa: F401

from hbwm.bdh.core import HBWMCore
from hbwm.bdh.upstream.bdh import Attention  # noqa: F401
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
