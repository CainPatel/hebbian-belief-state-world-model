import torch

from hbwm.baselines.lstm import LSTMConfig, LSTMLM
from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.instrument.recorder import SigmaRecorder, StateRecorder, make_recorder

TINY = HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34, dropout=0.0, block_size=64)


def test_sigma_recorder_fires_at_positions_and_matches_manual():
    torch.manual_seed(0)
    m = HBWMCore(TINY).eval()
    toks = torch.randint(0, 34, (3, 20))
    seen = []
    def fn(pos, payload):
        seen.append((pos, payload["sigma"].clone(), payload["x_sparse"].shape))
    final = SigmaRecorder(m).run(toks, [4, 9], fn)
    assert [s[0] for s in seen] == [4, 9] and final.t == 10  # stopped after last requested position
    st = m.init_state(3)
    with torch.no_grad():
        for t in range(10):
            _, st, _ = m.step(toks[:, t], st)
            if t == 4:
                assert torch.allclose(seen[0][1], st.sigma)
    assert torch.allclose(seen[1][1], st.sigma) and seen[0][2] == (2, 3, 2, 64)


def test_sigma_recorder_all_positions():
    m = HBWMCore(TINY).eval()
    n = []
    SigmaRecorder(m).run(torch.randint(0, 34, (2, 7)), None, lambda p, pl: n.append(p))
    assert n == list(range(7))


def test_state_recorder_lstm_and_factory():
    m = LSTMLM(LSTMConfig(vocab_size=34, n_embd=8, hidden=12, n_layer=2)).eval()
    assert isinstance(make_recorder(m), StateRecorder) and isinstance(make_recorder(HBWMCore(TINY)), SigmaRecorder)
    got = {}
    StateRecorder(m).run(torch.randint(0, 34, (4, 9)), [8], lambda p, pl: got.update({p: pl["state_vec"].shape}))
    assert got == {8: (4, 48)}
