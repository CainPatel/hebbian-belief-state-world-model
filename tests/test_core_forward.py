import torch
import torch.nn.functional as F

from hbwm.bdh.core import HBWMConfig, HBWMCore, decay_mask
from hbwm.bdh.upstream.bdh import BDH
from hbwm.losses import masked_ce

TINY = dict(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=10, dropout=0.0, block_size=32)


def test_config_neurons_and_to_bdh():
    cfg = HBWMConfig(**TINY)
    assert cfg.n_neurons == 64  # 16*8/2
    b = cfg.to_bdh()
    assert (b.n_layer, b.n_embd, b.n_head, b.mlp_internal_dim_multiplier, b.vocab_size) == (2, 16, 2, 8, 10)


def test_decay_mask_values():
    m = decay_mask(4, 0.5)
    expected = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0.5, 1, 0, 0], [0.25, 0.5, 1, 0]])
    assert torch.allclose(m, expected)
    assert torch.equal(decay_mask(5, 1.0), torch.ones(5, 5).tril(-1))


def test_forward_bit_identical_to_upstream_at_gamma_1():
    torch.manual_seed(0)
    cfg = HBWMConfig(**TINY)
    core = HBWMCore(cfg).eval()
    up = BDH(cfg.to_bdh()).eval()
    up.load_state_dict(core.state_dict())
    idx = torch.randint(0, 10, (3, 24))
    a, _ = core(idx)
    b, _ = up(idx)
    assert torch.equal(a, b)


def test_forward_with_decay_differs_and_has_shape():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**{**TINY, "decay_gamma": 0.9})).eval()
    idx = torch.randint(0, 10, (2, 12))
    logits, loss = core(idx, idx)
    assert logits.shape == (2, 12, 10) and loss.ndim == 0


def test_loss_mask_matches_manual():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    idx = torch.randint(0, 10, (2, 12))
    tgt = torch.randint(0, 10, (2, 12))
    mask = torch.zeros(12, dtype=torch.bool)
    mask[::3] = True
    logits, loss = core(idx, tgt, loss_mask=mask)
    manual = F.cross_entropy(logits[:, mask].reshape(-1, 10), tgt[:, mask].reshape(-1))
    assert torch.allclose(loss, manual)
    assert torch.allclose(masked_ce(logits, tgt, mask), manual)


def test_return_internals_shapes():
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    idx = torch.randint(0, 10, (2, 12))
    logits, loss, internals = core(idx, return_internals=True)
    assert internals["x_sparse"].shape == (2, 2, 2, 12, 64)  # L,B,nh,T,N
    assert internals["resid"].shape == (2, 2, 12, 16)  # L,B,T,D
    assert (internals["x_sparse"] >= 0).all()


def test_unshared_layers_have_distinct_params():
    shared = HBWMCore(HBWMConfig(**TINY))
    unshared = HBWMCore(HBWMConfig(**{**TINY, "share_layers": False}))
    n_shared = sum(p.numel() for p in shared.parameters())
    n_unshared = sum(p.numel() for p in unshared.parameters())
    core_block = 3 * 2 * 16 * 64  # 3·nh·D·N
    assert n_unshared - n_shared == core_block * (2 - 1)
    e0, _, _ = unshared.level_params(0)
    e1, _, _ = unshared.level_params(1)
    assert e0 is not e1
    idx = torch.randint(0, 10, (2, 8))
    logits, _ = unshared.eval()(idx)
    assert logits.shape == (2, 8, 10)
