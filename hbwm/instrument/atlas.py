import json
from pathlib import Path

import numpy as np
import torch

from hbwm.bdh.core import HBWMCore
from hbwm.device import release_memory
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.recorder import SigmaRecorder


@torch.no_grad()
def build_atlas(model: HBWMCore, data: EpisodeData, n_episodes=500, top_m=32, batch_eps=50, device=None,
                return_means=False):
    """Per level: top-m neurons (per head) by mean x_sparse when (a) a given token is read, (b) the agent
    stands on a given cell at observation end."""
    device = device if device is not None else next(model.parameters()).device
    c = model.hcfg
    L, nh, N, V, G = c.n_layer, c.n_head, c.n_neurons, c.vocab_size, data.G
    tok_sum = torch.zeros(L, V, nh, N)
    tok_cnt = torch.zeros(V)
    cell_sum = torch.zeros(L, G * G, nh, N)
    cell_cnt = torch.zeros(G * G)
    obs_pos = tk.obs_positions(data.L)
    obs_set = {int(p): t for t, p in enumerate(obs_pos)}
    n = min(n_episodes, data.n)
    rec = SigmaRecorder(model)
    for b0 in range(0, n, batch_eps):
        eps = np.arange(b0, min(n, b0 + batch_eps))
        tokens = torch.from_numpy(data.tokens[eps].astype(np.int64)).to(device)
        cells = torch.from_numpy((data.agent_pos[eps, :, 1] * G + data.agent_pos[eps, :, 0]).astype(np.int64))

        def fn(pos, payload):
            xs = payload["x_sparse"].float().cpu()  # L,B,nh,N
            toks = tokens[:, pos].cpu()
            tok_sum.index_add_(1, toks, xs)
            tok_cnt.index_add_(0, toks, torch.ones(len(toks)))
            if pos in obs_set:
                cc = cells[:, obs_set[pos]]
                cell_sum.index_add_(1, cc, xs)
                cell_cnt.index_add_(0, cc, torch.ones(len(cc)))

        rec.run(tokens, None, fn)
        # The atlas is the last recorder pass of a probe run, on a process that has already streamed
        # features several times; this pass touches every position of every episode, so return the
        # per-position temporaries to the system between batches rather than at the end.
        release_memory(device)
    tok_mean = tok_sum / tok_cnt.clamp(min=1)[None, :, None, None]
    cell_mean = cell_sum / cell_cnt.clamp(min=1)[None, :, None, None]
    atlas = {
        "top_m": top_m, "G": G, "n_levels": L, "n_head": nh,
        "token": tok_mean.topk(top_m, dim=-1).indices.tolist(),
        "cell": cell_mean.topk(top_m, dim=-1).indices.tolist(),
        "token_counts": tok_cnt.long().tolist(),
        "cell_counts": cell_cnt.long().tolist(),
    }
    # spec 4.8 measurement 5 needs the full [L, V, nh, N] profile, which atlas.json does not carry.
    return (atlas, tok_mean) if return_means else atlas


def save_atlas(atlas: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(atlas))


def load_atlas(path) -> dict:
    a = json.loads(Path(path).read_text())
    a["token"] = np.asarray(a["token"], dtype=np.int64)
    a["cell"] = np.asarray(a["cell"], dtype=np.int64)
    return a
