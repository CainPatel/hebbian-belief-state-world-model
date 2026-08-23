import numpy as np
import torch

from hbwm.baselines.lstm import LSTMConfig, LSTMLM
from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.instrument.features import BDH_FEATURES, extract, feature_dim, n_levels, neuron_of_feature
from hbwm.instrument.recorder import SigmaRecorder, StateRecorder

CFG = HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34, dropout=0.0, block_size=64)


def test_dims_and_extract_shapes():
    m = HBWMCore(CFG).eval()
    assert feature_dim(m, "sigma_full") == 2 * 64 * 16 and feature_dim(m, "sigma_rownorm") == 128
    assert feature_dim(m, "x_sparse") == 128 and feature_dim(m, "resid") == 16 and n_levels(m) == 2
    got = {}
    SigmaRecorder(m).run(torch.randint(0, 34, (3, 5)), [4], lambda p, pl: got.update({n: extract(pl, n, 1) for n in BDH_FEATURES}))
    for n in BDH_FEATURES:
        assert got[n].shape == (3, feature_dim(m, n))
    l = LSTMLM(LSTMConfig(vocab_size=34, n_embd=8, hidden=12, n_layer=2)).eval()
    assert feature_dim(l, "state_vec") == 48 and n_levels(l) == 1
    StateRecorder(l).run(torch.randint(0, 34, (2, 3)), [2], lambda p, pl: got.update({"sv": extract(pl, "state_vec", None)}))
    assert got["sv"].shape == (2, 48)


def test_sigma_full_layout_matches_neuron_of_feature():
    m = HBWMCore(CFG).eval()
    st = m.init_state(1)
    with torch.no_grad():
        for t in range(4):
            _, st, _ = m.step(torch.tensor([t + 1]), st)
    flat = extract({"sigma": st.sigma}, "sigma_full", 0)[0]
    h, n, d = 1, 37, 5
    f = h * 64 * 16 + n * 16 + d
    assert flat[f] == st.sigma[0, 0, h, n, d]
    assert neuron_of_feature(CFG, "sigma_full", np.array([f])).tolist() == [h * 64 + n]
    assert neuron_of_feature(CFG, "sigma_rownorm", np.array([70])).tolist() == [70]
