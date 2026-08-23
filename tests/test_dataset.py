import json
from pathlib import Path

import numpy as np
import torch

from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData, split_seed_ranges


def test_seed_ranges_disjoint(tiny_data):
    r = split_seed_ranges(tiny_data)
    spans = sorted(r.values())
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert e0 <= s1
    assert r["model_train"] == (0, 16) and r["probe_test"] == (44, 52)


def test_files_and_meta(tiny_data):
    d = Path(tiny_data.out_dir)
    assert all((d / f"{s}.npz").exists() for s in tiny_data.splits)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["T"] == tk.seq_len(8) and meta["vocab_size"] == 34 and meta["grid"]["size"] == 5


def test_loader_shapes_and_consistency(tiny_data):
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    assert d.n == 12 and d.T == 108 and d.L == 8 and d.G == 5 and d.n_obj == 2
    assert d.tokens.shape == (12, 108) and d.tokens.dtype == np.int16
    assert d.loss_mask.sum() == 11 * 9 and d.window_mask.sum() == 9 * 9
    for i in range(d.n):
        a, p, w = tk.decode_tokens(d.tokens[i])
        assert (p == d.agent_pos[i]).all()
    cheb = np.abs(d.obj_pos - d.agent_pos[:, :, None, :]).max(axis=3)
    assert (d.visible == (cheb <= 1)).all()
    assert ((d.steps_since_seen == 0) == d.visible).all()
    assert d.stale.shape == d.visible.shape and d.seed.tolist() == list(range(24, 36))


def test_get_batch(tiny_data):
    d = EpisodeData(tiny_data.out_dir, "model_train")
    x, y, m = d.get_batch(np.random.default_rng(0), 4, torch.device("cpu"))
    assert x.shape == (4, 107) and y.shape == (4, 107) and m.shape == (107,)
    assert x.dtype == torch.long and m.dtype == torch.bool
    assert torch.equal(x[:, 1:], y[:, :-1])
    x2, _, _ = d.batch_at([0, 1], torch.device("cpu"))
    assert x2.shape == (2, 107)
