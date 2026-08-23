import pytest
import torch

from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.bdh.upstream.bdh import Attention

TINY = dict(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=10, dropout=0.0, block_size=32)


def _rollout(core, idx, **kw):
    B, T = idx.shape
    state = core.init_state(B)
    outs, internals = [], []
    for t in range(T):
        logits, state, inner = core.step(idx[:, t], state, **kw)
        outs.append(logits)
        internals.append(inner)
    return torch.stack(outs, 1), state, internals


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_step_matches_forward(gamma):
    torch.manual_seed(1)
    core = HBWMCore(HBWMConfig(**{**TINY, "decay_gamma": gamma})).eval()
    idx = torch.randint(0, 10, (3, 24))
    with torch.no_grad():
        par, _ = core(idx)
        seq, state, _ = _rollout(core, idx)
    assert state.t == 24
    assert torch.allclose(par, seq, atol=1e-4), (par - seq).abs().max()


def test_sigma_closed_form():
    torch.manual_seed(2)
    gamma = 0.8
    core = HBWMCore(HBWMConfig(**{**TINY, "decay_gamma": gamma})).eval()
    idx = torch.randint(0, 10, (2, 10))
    with torch.no_grad():
        _, state, internals = _rollout(core, idx)
    T = 10
    for level in range(2):
        expected = torch.zeros_like(state.sigma[level])
        for s in range(T):
            xs = internals[s]["x_sparse"][level]  # B,nh,N
            k = Attention.rope((float(s) * core.attn.freqs).view(1, 1, -1), xs)
            v = internals[s]["resid"][level]  # B,D
            expected += (gamma ** (T - 1 - s)) * torch.einsum("bhn,bd->bhnd", k, v)
        assert torch.allclose(state.sigma[level], expected, atol=1e-5)


def test_plasticity_modes():
    torch.manual_seed(3)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    idx = torch.randint(0, 10, (2, 6))
    with torch.no_grad():
        _, st_full, _ = _rollout(core, idx)
        before = core.init_state(2)
        _, st_frozen, _ = _rollout(core, idx, plasticity="frozen")
        assert torch.equal(st_frozen.sigma, before.sigma)  # bit-identical zeros
        # scaled: one step from an arbitrary state, delta must be s x full delta
        base = st_full.clone()
        s_full = base.clone()
        core.step(idx[:, 0], s_full, plasticity="full")
        s_scaled = base.clone()
        core.step(idx[:, 0], s_scaled, plasticity="scaled", plasticity_scale=0.25)
        assert torch.allclose(s_scaled.sigma - base.sigma, 0.25 * (s_full.sigma - base.sigma), atol=1e-6)


def test_step_internals_shapes():
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    state = core.init_state(3)
    logits, state, inner = core.step(torch.tensor([1, 2, 3]), state)
    assert logits.shape == (3, 10)
    assert inner["x_sparse"].shape == (2, 3, 2, 64)
    assert inner["resid"].shape == (2, 3, 16)
    assert inner["yKV"].shape == (2, 3, 2, 16)
    assert state.sigma.shape == (2, 3, 2, 64, 16)
