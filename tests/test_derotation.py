import torch

from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.bdh.upstream.bdh import Attention
from hbwm.probes.structured import DerotProbe, FlatLinearProbe, StateShape, derotate

TINY = dict(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34,
            dropout=0.0, block_size=64)
SHAPE = StateShape(n_heads=2, rows=64, cols=16, rotary=True)


def freqs_of(core):
    return core.attn.freqs.reshape(-1)


def test_derotate_is_invertible():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    f = freqs_of(core)
    sigma = torch.randn(3, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    t = torch.tensor([5.0, 17.0, 40.0])
    back = derotate(derotate(sigma, t, f), -t, f)
    assert torch.allclose(back, sigma, atol=1e-5)


def test_derotate_turns_absolute_phase_into_relative_phase():
    """A single write at step s, read at step t, derotates to phase s - t (spec Appendix A)."""
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    f = freqs_of(core)
    u = torch.rand(1, SHAPE.n_heads, SHAPE.rows)  # relu output: nonnegative
    x = torch.randn(1, SHAPE.cols)
    s, t = 9.0, 23.0
    k_s = Attention.rope((s * f).view(1, 1, -1), u)
    sigma = torch.einsum("bhn,bd->bhnd", k_s, x)
    want_key = Attention.rope(((s - t) * f).view(1, 1, -1), u)
    want = torch.einsum("bhn,bd->bhnd", want_key, x)
    assert torch.allclose(derotate(sigma, torch.tensor([t]), f), want, atol=1e-5)


def test_derot_probe_is_not_equal_to_the_plain_probe_on_mixed_positions():
    """The transform depends on the example's own t, so no fixed W absorbs it (spec 4.4)."""
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    inner = FlatLinearProbe(SHAPE, 5)
    p = DerotProbe(inner, SHAPE, freqs_of(core))
    x = torch.randn(4, SHAPE.n_features)
    t = torch.tensor([0.0, 11.0, 26.0, 47.0])
    assert not torch.allclose(p(x, t), inner(x), atol=1e-3)


def test_derot_probe_at_position_zero_is_the_plain_probe():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    inner = FlatLinearProbe(SHAPE, 5)
    p = DerotProbe(inner, SHAPE, freqs_of(core))
    x = torch.randn(4, SHAPE.n_features)
    assert torch.allclose(p(x, torch.zeros(4)), inner(x), atol=1e-5)


def test_set_positions_feeds_the_forward_that_takes_no_t():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    p = DerotProbe(FlatLinearProbe(SHAPE, 5), SHAPE, freqs_of(core))
    x = torch.randn(4, SHAPE.n_features)
    t = torch.tensor([3.0, 8.0, 14.0, 30.0])
    p.set_positions(t)
    assert torch.allclose(p(x), p(x, t), atol=1e-6)


def _standardize_then_derotate(inner, x, t, mean, std, freqs):
    """The REJECTED order, built here only so the test can show it differs (spec 4.4)."""
    z = ((x - mean) / std).view(-1, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    return inner(derotate(z, t, freqs).reshape(x.shape[0], -1))


def test_the_two_orders_differ_under_per_entry_scales():
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    f = freqs_of(core)
    mean = torch.zeros(SHAPE.n_features)
    std = torch.rand(SHAPE.n_features) + 0.5  # a different scale for every (h, n, d)
    inner = FlatLinearProbe(SHAPE, 5)
    p = DerotProbe(inner, SHAPE, f, mean, std)  # derotate, then standardize: the preregistered order
    x = torch.randn(4, SHAPE.n_features)
    t = torch.tensor([3.0, 8.0, 14.0, 30.0])
    assert not torch.allclose(p(x, t), _standardize_then_derotate(inner, x, t, mean, std, f), atol=1e-4)


def test_the_two_orders_agree_under_pair_shared_scales():
    """The documented fallback: one scale per rotated pair makes scaling commute with the rotation."""
    torch.manual_seed(0)
    core = HBWMCore(HBWMConfig(**TINY)).eval()
    f = freqs_of(core)
    pair = torch.rand(SHAPE.rows // 2) + 0.5
    row_scale = pair.repeat_interleave(2)  # equal within each rotated pair (2j, 2j+1)
    std = (row_scale.view(1, SHAPE.rows, 1)
           .expand(SHAPE.n_heads, SHAPE.rows, SHAPE.cols).reshape(-1).contiguous())
    mean = torch.zeros(SHAPE.n_features)
    inner = FlatLinearProbe(SHAPE, 5)
    p = DerotProbe(inner, SHAPE, f, mean, std)
    x = torch.randn(4, SHAPE.n_features)
    t = torch.tensor([3.0, 8.0, 14.0, 30.0])
    assert torch.allclose(p(x, t), _standardize_then_derotate(inner, x, t, mean, std, f), atol=1e-5)
