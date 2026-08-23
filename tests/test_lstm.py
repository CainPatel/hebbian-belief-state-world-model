import torch

from hbwm.baselines.lstm import LSTMLM, LSTMConfig

TINY = LSTMConfig(vocab_size=34, n_embd=8, hidden=12, n_layer=2, dropout=0.0)


def test_forward_and_state_dim():
    m = LSTMLM(TINY).eval()
    logits, loss = m(torch.randint(0, 34, (2, 10)), torch.randint(0, 34, (2, 10)))
    assert logits.shape == (2, 10, 34) and loss.ndim == 0
    assert m.state_dim == 2 * 2 * 12


def test_step_matches_forward():
    torch.manual_seed(0)
    m = LSTMLM(TINY).eval()
    idx = torch.randint(0, 34, (3, 15))
    with torch.no_grad():
        par, _ = m(idx)
        st = m.init_state(3)
        outs = []
        for t in range(15):
            lg, st = m.step(idx[:, t], st)
            outs.append(lg)
        seq = torch.stack(outs, 1)
    assert torch.allclose(par, seq, atol=1e-5)
    assert m.state_vector(st).shape == (3, m.state_dim)


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = LSTMLM(TINY)
    x = torch.randint(0, 34, (2, 20))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        _, loss = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
