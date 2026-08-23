import pytest
import torch

from hbwm.baselines.rwkv import RWKVLM, RWKVConfig, wkv_chunked, wkv_sequential, wkv_step

TINY = RWKVConfig(vocab_size=34, n_embd=16, n_layer=2, chunk_size=5)


def test_wkv_step_matches_sequential():
    torch.manual_seed(0)
    B, T, C = 2, 9, 6
    w, u = -torch.exp(torch.randn(C)), torch.randn(C)
    k, v = 3 * torch.randn(B, T, C), torch.randn(B, T, C)
    ref = wkv_sequential(w, u, k, v)
    aa, bb, pp = torch.zeros(B, C), torch.zeros(B, C), torch.full((B, C), float("-inf"))
    outs = []
    for t in range(T):
        o, aa, bb, pp = wkv_step(w, u, k[:, t], v[:, t], aa, bb, pp)
        outs.append(o)
    assert torch.allclose(torch.stack(outs, 1), ref, atol=1e-6)


def test_wkv_matches_naive_reference():
    torch.manual_seed(0)
    B, T, C = 2, 11, 5
    w = -torch.exp(torch.randn(C, dtype=torch.float64))
    u = torch.randn(C, dtype=torch.float64)
    k = 2 * torch.randn(B, T, C, dtype=torch.float64)
    v = torch.randn(B, T, C, dtype=torch.float64)
    outs = []
    for t in range(T):
        num = sum(torch.exp(k[:, i] + (t - 1 - i) * w) * v[:, i] for i in range(t)) + torch.exp(u + k[:, t]) * v[:, t]
        den = sum(torch.exp(k[:, i] + (t - 1 - i) * w) for i in range(t)) + torch.exp(u + k[:, t])
        outs.append(num / den)
    ref = torch.stack(outs, 1)
    assert torch.allclose(wkv_sequential(w, u, k, v), ref, atol=1e-12)
    assert torch.allclose(wkv_chunked(w, u, k, v, 4), ref, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_wkv_chunked_matches_sequential_values_and_grads(chunk):
    torch.manual_seed(1)
    B, T, C = 2, 150, 5
    w, u = -torch.exp(torch.randn(C)), torch.randn(C)
    k1 = (3 * torch.randn(B, T, C)).requires_grad_(True)
    v1 = torch.randn(B, T, C).requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    ref = wkv_sequential(w, u, k1, v1)
    got = wkv_chunked(w, u, k2, v2, chunk)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    g = torch.randn_like(ref)
    (ref * g).sum().backward()
    (got * g).sum().backward()
    assert torch.allclose(k1.grad, k2.grad, atol=1e-4) and torch.allclose(v1.grad, v2.grad, atol=1e-4)


def test_model_forward_uses_chunked_and_matches_step():
    torch.manual_seed(0)
    m = RWKVLM(RWKVConfig(vocab_size=34, n_embd=16, n_layer=2, chunk_size=5)).eval()
    idx = torch.randint(0, 34, (2, 33))
    with torch.no_grad():
        par, _ = m(idx)
        st = m.init_state(2)
        outs = []
        for t in range(33):
            lg, st = m.step(idx[:, t], st)
            outs.append(lg)
    assert torch.allclose(par, torch.stack(outs, 1), atol=1e-4)


def test_forward_matches_step_and_state_dim():
    torch.manual_seed(0)
    m = RWKVLM(TINY).eval()
    idx = torch.randint(0, 34, (3, 23))
    with torch.no_grad():
        par, _ = m(idx)
        st = m.init_state(3)
        outs = []
        for t in range(23):
            lg, st = m.step(idx[:, t], st)
            outs.append(lg)
        seq = torch.stack(outs, 1)
    assert torch.allclose(par, seq, atol=1e-4), (par - seq).abs().max()
    assert m.state_dim == 5 * 16 * 2 and m.state_vector(st).shape == (3, m.state_dim)
    assert torch.isfinite(m.state_vector(st)).all()


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = RWKVLM(TINY)
    x = torch.randint(0, 34, (2, 20))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        _, loss = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
