"""Fixed 34-token vocabulary for (action, observation) sequences. Sequence: BOS, o_0, a_1, o_1, ..., a_L, o_L."""

import numpy as np

BOS, PAD = 0, 1
A_BASE, N_ACTIONS = 2, 4
X_BASE, Y_BASE, MAX_G = 6, 17, 11
CELL_BASE = 28  # token = CELL_BASE + cell code: EMPTY(0)->28, WALL(1)->29, OBJ k (2+k)->30+k
EMPTY_TOK, WALL_TOK, OBJ_BASE = 28, 29, 30
VOCAB_SIZE = 34
OBS_LEN = 11  # X, Y, 9 window cells
STEP_LEN = 12  # action + observation


def seq_len(L: int) -> int:
    return 1 + OBS_LEN + STEP_LEN * L


def obs_positions(L: int) -> np.ndarray:
    """Index of the last token of o_t for t = 0..L."""
    return OBS_LEN + STEP_LEN * np.arange(L + 1)


def action_positions(L: int) -> np.ndarray:
    """Index of a_t for t = 1..L."""
    return STEP_LEN * np.arange(1, L + 1)


def encode_observation(x: int, y: int, window9) -> np.ndarray:
    out = np.empty(OBS_LEN, dtype=np.int16)
    out[0] = X_BASE + int(x)
    out[1] = Y_BASE + int(y)
    out[2:] = CELL_BASE + np.asarray(window9, dtype=np.int16)
    return out


def encode_episode(actions, agent_pos, windows) -> np.ndarray:
    L = len(actions)
    toks = np.empty(seq_len(L), dtype=np.int16)
    toks[0] = BOS
    toks[1 : 1 + OBS_LEN] = encode_observation(agent_pos[0, 0], agent_pos[0, 1], windows[0])
    for t in range(1, L + 1):
        base = STEP_LEN * t
        toks[base] = A_BASE + int(actions[t - 1])
        toks[base + 1 : base + 1 + OBS_LEN] = encode_observation(agent_pos[t, 0], agent_pos[t, 1], windows[t])
    return toks


def decode_tokens(tokens):
    toks = np.asarray(tokens, dtype=np.int64)
    L = (len(toks) - 1 - OBS_LEN) // STEP_LEN
    assert len(toks) == seq_len(L) and toks[0] == BOS
    actions = toks[action_positions(L)] - A_BASE
    ends = obs_positions(L)
    obs = np.stack([toks[e - OBS_LEN + 1 : e + 1] for e in ends])  # [L+1, 11]
    agent_pos = np.stack([obs[:, 0] - X_BASE, obs[:, 1] - Y_BASE], axis=1)
    windows = obs[:, 2:] - CELL_BASE
    return actions, agent_pos, windows


def loss_mask(L: int) -> np.ndarray:
    m = np.ones(seq_len(L), dtype=bool)
    m[0] = False
    m[action_positions(L)] = False
    return m


def window_mask(L: int) -> np.ndarray:
    m = np.zeros(seq_len(L), dtype=bool)
    for e in obs_positions(L):
        m[e - 8 : e + 1] = True
    return m


def token_name(tok: int) -> str:
    tok = int(tok)
    if tok == BOS:
        return "BOS"
    if tok == PAD:
        return "PAD"
    if A_BASE <= tok < A_BASE + N_ACTIONS:
        return "A_" + "NESW"[tok - A_BASE]
    if X_BASE <= tok < X_BASE + MAX_G:
        return f"X_{tok - X_BASE}"
    if Y_BASE <= tok < Y_BASE + MAX_G:
        return f"Y_{tok - Y_BASE}"
    if tok == EMPTY_TOK:
        return "EMPTY"
    if tok == WALL_TOK:
        return "WALL"
    if OBJ_BASE <= tok < VOCAB_SIZE:
        return f"OBJ_{tok - OBJ_BASE}"
    raise ValueError(tok)
