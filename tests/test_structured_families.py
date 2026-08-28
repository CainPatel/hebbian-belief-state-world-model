import numpy as np
import pytest
import torch

from hbwm.probes.structured import (
    MLP_HIDDEN,
    STUDY2_RESTARTS,
    DerotProbe,
    FamilySpec,
    FlatLinearProbe,
    MLPProbe,
    QueryRankProbe,
    StateShape,
    apply_randproj,
    build_family,
    family_specs,
    param_count,
    sparse_randproj,
    spec_label,
)
from hbwm.probes.structured import (
    QueryRankProbe as _QRP,
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


def test_mlp_probe_shapes_and_hidden_width():
    p = MLPProbe(64, C)
    assert p(torch.randn(7, 64)).shape == (7, C)
    assert p.net[0].out_features == MLP_HIDDEN


def test_mlp_probe_is_nonlinear():
    # A zero-bias ReLU net is positively homogeneous, so scaling by +2 would NOT separate it from a
    # linear map. Negative scaling does: relu(-z) != -relu(z).
    torch.manual_seed(0)
    p = MLPProbe(8, C, hidden=16)
    x = torch.randn(4, 8)
    assert not torch.allclose(p(-x) - p.bias, -(p(x) - p.bias), atol=1e-4)


def test_sparse_randproj_is_deterministic_and_correctly_shaped():
    idx, sign = sparse_randproj(1000, 32, density=8, seed=7)
    idx2, sign2 = sparse_randproj(1000, 32, density=8, seed=7)
    assert idx.shape == (32, 8) and sign.shape == (32, 8)
    assert np.array_equal(idx, idx2) and np.array_equal(sign, sign2)
    assert idx.min() >= 0 and idx.max() < 1000
    assert set(np.unique(sign)) <= {-1.0, 1.0}
    assert all(len(np.unique(row)) == 8 for row in idx)  # sampled without replacement


def test_apply_randproj_matches_the_explicit_dense_equivalent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 40)).astype(np.float32)
    idx, sign = sparse_randproj(40, 6, density=4, seed=1)
    dense = np.zeros((40, 6), dtype=np.float32)
    for j in range(6):
        dense[idx[j], j] = sign[j]
    assert np.allclose(apply_randproj(x, idx, sign), x @ dense, atol=1e-5)


def test_readout_parameters_excludes_every_bias_by_name():
    # spec 4.7 item 1: the L2 penalty is weights only. Assert on parameter NAMES, not just counts,
    # so a future rename of a weight or bias tensor can't silently re-admit a bias into the penalty.
    flat = FlatLinearProbe(SHAPE, C)
    ids = {id(p) for p in flat.readout_parameters()}
    assert {n for n, p in flat.named_parameters() if id(p) in ids} == {"weight"}

    qr = QueryRankProbe(SHAPE, C, rank=2)
    ids = {id(p) for p in qr.readout_parameters()}
    assert {n for n, p in qr.named_parameters() if id(p) in ids} == {"q", "v"}

    mlp = MLPProbe(64, C)
    ids = {id(p) for p in mlp.readout_parameters()}
    assert {n for n, p in mlp.named_parameters() if id(p) in ids} == {"net.0.weight", "net.2.weight"}


def test_mlp_probe_weight_only_count_matches_param_count():
    p = MLPProbe(64, C)
    learned = sum(t.numel() for n, t in p.named_parameters() if not n.endswith("bias"))
    assert learned == param_count("mlp_state", SHAPE, C, n_input=64)


def test_shared_query_applies_one_identical_query_bank_to_every_class():
    torch.manual_seed(0)
    p = QueryRankProbe(SHAPE, C, rank=2, shared_query=True)
    W = p.flat_weight()
    for c in range(C):
        expected = torch.einsum("hjp,hjq->hpq", p.q, p.v[c])
        assert torch.allclose(W[c], expected, atol=1e-6)


BDH_REF = StateShape(n_heads=4, rows=2048, cols=64, rotary=True)
LSTM_REF = StateShape(n_heads=1, rows=4, cols=350, rotary=False)


def test_family_specs_for_a_rotary_state():
    labels = [spec_label(s) for s in family_specs(BDH_REF)]
    assert labels == [
        "flat_linear", "query_rank_1", "query_rank_4", "query_rank_16",
        "shared_query_rank_1", "shared_query_rank_4", "shared_query_rank_16",
        "derot_flat_linear", "derot_query_rank_1", "derot_query_rank_4", "derot_query_rank_16",
        "mlp_state", "mlp_rownorm", "mlp_randproj",
    ]


def test_mlp_state_is_family_5s_matched_arm_on_every_state():
    for shape in (BDH_REF, LSTM_REF, StateShape(n_heads=1, rows=20, cols=176, rotary=False)):
        labels = [spec_label(s) for s in family_specs(shape)]
        assert "mlp_state" in labels
    bdh_only = {"mlp_rownorm", "mlp_randproj"}
    assert bdh_only <= set(spec_label(s) for s in family_specs(BDH_REF))
    assert not (bdh_only & set(spec_label(s) for s in family_specs(LSTM_REF)))


def test_family_specs_for_a_baseline_state_drop_the_derot_arms():
    labels = [spec_label(s) for s in family_specs(LSTM_REF)]
    assert labels == [
        "flat_linear", "query_rank_1", "query_rank_4", "query_rank_16",
        "shared_query_rank_1", "shared_query_rank_4", "shared_query_rank_16", "mlp_state",
    ]
    assert not any(lbl.startswith("derot") for lbl in labels)


def test_restarts_are_three_for_factorized_families_and_one_otherwise():
    got = {spec_label(s): s.n_restarts for s in family_specs(BDH_REF)}
    assert got["query_rank_4"] == 3 and got["shared_query_rank_4"] == 3
    assert got["derot_query_rank_4"] == 3
    assert got["flat_linear"] == 1 and got["derot_flat_linear"] == 1
    assert got["mlp_state"] == 1 and got["mlp_rownorm"] == 1


def test_matched_family_contract_every_family_builds_on_every_state():
    tiny_bdh = StateShape(n_heads=2, rows=8, cols=4, rotary=True)
    tiny_lstm = StateShape(n_heads=1, rows=4, cols=6, rotary=False)
    freqs = torch.rand(tiny_bdh.rows)
    for shape, f in ((tiny_bdh, freqs), (tiny_lstm, None)):
        for spec in family_specs(shape):
            n_in = {"flat": shape.n_features, "state": shape.n_features,
                    "rownorm": shape.n_heads * shape.rows, "randproj": 12}[spec.input_kind]
            p = build_family(spec, shape, C, n_in, freqs=f)
            x = torch.randn(3, n_in)
            p.set_positions(torch.zeros(3))
            assert p(x).shape == (3, C), spec_label(spec)


def test_derot_families_wrap_only_on_a_rotary_state():
    tiny = StateShape(n_heads=2, rows=8, cols=4, rotary=True)
    spec = next(s for s in family_specs(tiny) if spec_label(s) == "derot_query_rank_4")
    p = build_family(spec, tiny, C, tiny.n_features, freqs=torch.rand(tiny.rows))
    assert isinstance(p, DerotProbe) and isinstance(p.inner, _QRP)


def test_derot_families_alias_the_plain_family_on_a_non_rotary_state():
    """Spec 5.1: on a state with no rotary phase, derot_* IS its undecorated counterpart."""
    tiny = StateShape(n_heads=1, rows=4, cols=6, rotary=False)
    p_flat = build_family(FamilySpec("derot_flat_linear"), tiny, C, tiny.n_features)
    assert not isinstance(p_flat, DerotProbe) and isinstance(p_flat, FlatLinearProbe)

    p_qr = build_family(
        FamilySpec("derot_query_rank_r", 2, n_restarts=STUDY2_RESTARTS), tiny, C, tiny.n_features
    )
    assert not isinstance(p_qr, DerotProbe) and isinstance(p_qr, _QRP)


def test_derot_flat_linear_alias_matches_the_plain_probe_bit_for_bit_on_a_baseline():
    """The alias is not merely the same class: same seed/mean/std must yield identical params."""
    tiny = StateShape(n_heads=1, rows=4, cols=6, rotary=False)
    mean = torch.randn(tiny.n_features)
    std = torch.rand(tiny.n_features) + 0.5

    gen_plain = torch.Generator().manual_seed(0)
    p_plain = build_family(FamilySpec("flat_linear"), tiny, C, tiny.n_features, mean, std, gen=gen_plain)

    gen_derot = torch.Generator().manual_seed(0)
    p_derot = build_family(
        FamilySpec("derot_flat_linear"), tiny, C, tiny.n_features, mean, std, gen=gen_derot
    )

    assert isinstance(p_derot, FlatLinearProbe) and not isinstance(p_derot, DerotProbe)
    assert torch.allclose(p_plain.weight, p_derot.weight)
    assert torch.allclose(p_plain.bias, p_derot.bias)


def test_derot_family_on_a_rotary_state_requires_freqs():
    tiny = StateShape(n_heads=2, rows=8, cols=4, rotary=True)
    spec = FamilySpec("derot_query_rank_r", 2, n_restarts=STUDY2_RESTARTS)
    with pytest.raises(ValueError):
        build_family(spec, tiny, C, tiny.n_features, freqs=None)
