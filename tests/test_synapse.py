import torch

from hbwm.bdh.core import HBWMConfig, HBWMCore

TINY = dict(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=10, dropout=0.0, block_size=32)


def test_synapse_equals_explicit_product():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    idx = torch.randint(0, 10, (2, 9))
    state = core.init_state(2)
    with torch.no_grad():
        for t in range(9):
            _, state, _ = core.step(idx[:, t], state)
    level, head = 1, 0
    rows = torch.tensor([3, 10, 50])
    cols = torch.tensor([0, 7])
    got = core.synapse(state.sigma[level], level, head, rows, cols)
    _, enc_v, _ = core.level_params(level)
    full = state.sigma[level][:, head] @ enc_v[head]  # B,N,N materialised
    assert got.shape == (2, 3, 2)
    assert torch.allclose(got, full[:, rows][:, :, cols], atol=1e-6)
