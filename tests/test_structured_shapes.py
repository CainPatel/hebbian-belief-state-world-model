import pytest

from hbwm.baselines.lstm import LSTMLM, LSTMConfig
from hbwm.baselines.rwkv import RWKVLM, RWKVConfig
from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.probes.structured import STUDY2_RANKS, StateShape, param_count, state_shape

# Reference config, spec section 4 notation: nh = 4, N = 2048, D = 64, C = 81 classes.
BDH_REF = StateShape(n_heads=4, rows=2048, cols=64, rotary=True)
LSTM_REF = StateShape(n_heads=1, rows=4, cols=350, rotary=False)
RWKV_REF = StateShape(n_heads=1, rows=20, cols=176, rotary=False)


def test_reference_feature_counts_match_study1():
    # Study 1 feature counts, RESULTS.md probe accuracy table.
    assert BDH_REF.n_features == 524288
    assert LSTM_REF.n_features == 1400
    assert RWKV_REF.n_features == 3520


def test_saturation_points_match_spec_5_2():
    assert (BDH_REF.saturation_rank, BDH_REF.shared_saturation_rank) == (64, 2048)
    assert (LSTM_REF.saturation_rank, LSTM_REF.shared_saturation_rank) == (4, 4)
    assert (RWKV_REF.saturation_rank, RWKV_REF.shared_saturation_rank) == (20, 20)


def test_rank_fractions_match_spec_5_2_table():
    got = {name: [round(s.rank_fraction(r), 2) for r in STUDY2_RANKS]
           for name, s in (("bdh", BDH_REF), ("lstm", LSTM_REF), ("rwkv", RWKV_REF))}
    assert got == {"bdh": [0.02, 0.06, 0.25], "lstm": [0.25, 1.0, 1.0], "rwkv": [0.05, 0.2, 0.8]}


def test_state_shape_from_models():
    bdh = HBWMCore(HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8,
                              vocab_size=34, dropout=0.0, block_size=64))
    assert state_shape(bdh) == StateShape(n_heads=2, rows=64, cols=16, rotary=True)

    lstm = LSTMLM(LSTMConfig(vocab_size=34, n_embd=8, hidden=6, n_layer=2))
    s = state_shape(lstm)
    assert s == StateShape(n_heads=1, rows=4, cols=6, rotary=False)
    assert s.n_features == lstm.state_dim

    rwkv = RWKVLM(RWKVConfig(vocab_size=34, n_embd=5, n_layer=3))
    s = state_shape(rwkv)
    assert s == StateShape(n_heads=1, rows=15, cols=5, rotary=False)
    assert s.n_features == rwkv.state_dim


def test_parameter_counts_match_spec_appendix_b():
    C = 81
    assert param_count("flat_linear", BDH_REF, C) == 42467328
    assert param_count("query_rank_r", BDH_REF, C, 1) == 684288
    assert param_count("query_rank_r", BDH_REF, C, 4) == 2737152
    assert param_count("query_rank_r", BDH_REF, C, 16) == 10948608
    assert param_count("shared_query_rank_r", BDH_REF, C, 1) == 28928
    assert param_count("shared_query_rank_r", BDH_REF, C, 16) == 462848
    assert param_count("flat_linear", LSTM_REF, C) == 113400
    assert param_count("query_rank_r", LSTM_REF, C, 1) == 28674
    assert param_count("flat_linear", RWKV_REF, C) == 285120
    assert param_count("query_rank_r", RWKV_REF, C, 1) == 15876


def test_mlp_parameter_counts_match_spec_appendix_b():
    C = 81
    # spec Appendix B: mlp_state is F * 512 + 512 * C for each state's own width.
    assert param_count("mlp_state", BDH_REF, C, n_input=BDH_REF.n_features) == 268476928
    assert param_count("mlp_state", LSTM_REF, C, n_input=LSTM_REF.n_features) == 758272
    assert param_count("mlp_state", RWKV_REF, C, n_input=RWKV_REF.n_features) == 1843712
    assert param_count("mlp_rownorm", BDH_REF, C, n_input=8192) == 4235776
    assert param_count("mlp_randproj", BDH_REF, C, n_input=4096) == 2138624


def test_rank_requires_a_rank_argument():
    with pytest.raises(ValueError):
        param_count("query_rank_r", BDH_REF, 81)
    with pytest.raises(ValueError):
        param_count("mlp_state", BDH_REF, 81)
