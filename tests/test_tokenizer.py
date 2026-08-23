import numpy as np

from hbwm.envs import tokenizer as tk


def test_constants_and_lengths():
    assert tk.VOCAB_SIZE == 34 and tk.seq_len(96) == 1164
    assert tk.obs_positions(2).tolist() == [11, 23, 35]
    assert tk.action_positions(3).tolist() == [12, 24, 36]


def test_masks():
    L = 5
    lm, wm = tk.loss_mask(L), tk.window_mask(L)
    assert lm.shape == (tk.seq_len(L),) and lm.sum() == 11 * (L + 1) and wm.sum() == 9 * (L + 1)
    assert not lm[0] and not lm[tk.action_positions(L)].any()
    assert lm[tk.obs_positions(L)].all() and wm[tk.obs_positions(L)].all()
    assert not wm[tk.obs_positions(L) - 9].any()  # X token is not a window token


def test_round_trip():
    rng = np.random.default_rng(0)
    L, G = 4, 9
    actions = rng.integers(0, 4, size=L)
    agent_pos = rng.integers(0, G, size=(L + 1, 2))
    windows = rng.integers(0, 6, size=(L + 1, 9))  # codes 0..5
    toks = tk.encode_episode(actions, agent_pos, windows)
    assert toks.dtype == np.int16 and toks.shape == (tk.seq_len(L),) and toks[0] == tk.BOS
    assert toks.max() < tk.VOCAB_SIZE and toks.min() >= 0
    a, p, w = tk.decode_tokens(toks)
    assert (a == actions).all() and (p == agent_pos).all() and (w == windows).all()


def test_token_names():
    assert tk.token_name(0) == "BOS" and tk.token_name(2) == "A_N" and tk.token_name(6) == "X_0"
    assert tk.token_name(17) == "Y_0" and tk.token_name(28) == "EMPTY" and tk.token_name(29) == "WALL"
    assert tk.token_name(30) == "OBJ_0" and tk.token_name(33) == "OBJ_3"
