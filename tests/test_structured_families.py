import torch

from hbwm.probes.structured import (
    FlatLinearProbe,
    QueryRankProbe,
    StateShape,
    param_count,
)

SHAPE = StateShape(n_heads=2, rows=8, cols=4, rotary=False)  # saturation_rank = 4
C = 3


def test_flat_linear_matches_an_explicit_contraction():
    torch.manual_seed(0)
    p = FlatLinearProbe(SHAPE, C)
    x = torch.randn(5, SHAPE.n_features)
    W = p.weight.detach().view(C, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    z = x.view(5, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    expected = torch.einsum("bhpq,chpq->bc", z, W) + p.bias.detach()
    assert torch.allclose(p(x), expected, atol=1e-5)


def test_rank_r_at_saturation_represents_any_linear_probe():
    """r >= min(P, Q) with unconstrained factors spans every flat weight (spec 5.2)."""
    torch.manual_seed(0)
    W = torch.randn(C, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    p = QueryRankProbe(SHAPE, C, rank=SHAPE.saturation_rank)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # [C,nh,P,r], [C,nh,r], [C,nh,r,Q]
    with torch.no_grad():
        p.q.copy_((U * S.unsqueeze(-2)).transpose(-1, -2))
        p.v.copy_(Vh)
        p.bias.zero_()
    assert torch.allclose(p.flat_weight(), W, atol=1e-5)
    x = torch.randn(5, SHAPE.n_features)
    z = x.view(5, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    assert torch.allclose(p(x), torch.einsum("bhpq,chpq->bc", z, W), atol=1e-5)


def test_rank_1_cannot_represent_a_full_rank_weight():
    torch.manual_seed(0)
    W = torch.randn(C, SHAPE.n_heads, SHAPE.rows, SHAPE.cols)
    p = QueryRankProbe(SHAPE, C, rank=1)
    assert not torch.allclose(p.flat_weight(), W, atol=1e-3)
    assert torch.linalg.matrix_rank(p.flat_weight()[0, 0]).item() == 1


def test_shared_query_shares_one_query_bank_across_classes():
    p = QueryRankProbe(SHAPE, C, rank=2, shared_query=True)
    assert tuple(p.q.shape) == (SHAPE.n_heads, 2, SHAPE.rows)
    assert tuple(p.v.shape) == (C, SHAPE.n_heads, 2, SHAPE.cols)


def test_module_parameter_counts_match_param_count():
    for family, kwargs in (("flat_linear", {}), ("query_rank_r", {"rank": 2}),
                           ("shared_query_rank_r", {"rank": 2, "shared_query": True})):
        p = FlatLinearProbe(SHAPE, C) if family == "flat_linear" else QueryRankProbe(SHAPE, C, **kwargs)
        learned = sum(t.numel() for n, t in p.named_parameters() if n != "bias")
        assert learned == param_count(family, SHAPE, C, kwargs.get("rank")), family


def test_standardization_buffers_are_applied():
    p = FlatLinearProbe(SHAPE, C, mean=torch.full((SHAPE.n_features,), 2.0),
                        std=torch.full((SHAPE.n_features,), 4.0))
    x = torch.full((1, SHAPE.n_features), 2.0)
    assert torch.allclose(p(x), p.bias.detach().unsqueeze(0), atol=1e-6)
