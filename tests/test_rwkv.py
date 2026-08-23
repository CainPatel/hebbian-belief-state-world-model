import torch

from hbwm.baselines.rwkv import RWKVConfig, RWKVLM, wkv_sequential, wkv_step

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
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]
