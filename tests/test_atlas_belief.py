import numpy as np
import torch

from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.atlas import build_atlas, load_atlas, save_atlas
from hbwm.instrument.belief import belief_map

CFG = HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34, dropout=0.0, block_size=128)


def test_atlas_shapes_and_roundtrip(tiny_data, tmp_path):
    m = HBWMCore(CFG).eval()
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    atlas = build_atlas(m, d, n_episodes=6, top_m=4, batch_eps=3)
    assert np.asarray(atlas["token"]).shape == (2, 34, 2, 4)
    assert np.asarray(atlas["cell"]).shape == (2, 25, 2, 4)
    assert np.asarray(atlas["token"]).max() < 64 and sum(atlas["token_counts"]) == 6 * d.T
    save_atlas(atlas, tmp_path / "a.json")
    back = load_atlas(tmp_path / "a.json")
    assert (back["token"] == np.asarray(atlas["token"])).all() and back["G"] == 5


def test_belief_map_matches_bruteforce(tiny_data):
    torch.manual_seed(0)
    m = HBWMCore(CFG).eval()
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    atlas = load_atlas_like(build_atlas(m, d, n_episodes=4, top_m=3, batch_eps=4))
    sigma = torch.randn(2, 64, 16)  # nh,N,D for one level
    level, obj = 1, 2
    G = 5
    got = belief_map(m, sigma, level, atlas, obj, G)
    _, enc_v, _ = m.level_params(level)
    tokrow = atlas["token"][level][tk.OBJ_BASE + obj]  # nh, m
    exp = np.zeros((G, G))
    for y in range(G):
        for x in range(G):
            c = y * G + x
            for h in range(2):
                rows = atlas["cell"][level][c][h]
                cols = tokrow[h]
                full = (sigma[h] @ enc_v[h]).detach().numpy()  # N x N
                exp[y, x] += full[np.ix_(rows, cols)].sum() + full[np.ix_(cols, rows)].sum()
    assert got.shape == (G, G) and np.allclose(got, exp, atol=1e-4)


def load_atlas_like(atlas):
    a = dict(atlas)
    a["token"], a["cell"] = np.asarray(atlas["token"]), np.asarray(atlas["cell"])
    return a
