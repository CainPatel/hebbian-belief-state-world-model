from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hbwm.envs import tokenizer as tk
from hbwm.instrument.features import extract
from hbwm.instrument.recorder import make_recorder


def iter_features(model, data, pairs, specs, batch_eps=64, device=None):
    """Yield (pair_indices, {spec: X[n_i, F]}) batches; each pair index is yielded exactly once."""
    device = device if device is not None else next(model.parameters()).device
    rec = make_recorder(model)
    obs_pos = tk.obs_positions(data.L)
    unique_eps = np.unique(pairs.ep)
    for b0 in range(0, len(unique_eps), batch_eps):
        eps = unique_eps[b0 : b0 + batch_eps]
        row_of_ep = {int(e): i for i, e in enumerate(eps)}
        idx_b = np.where(np.isin(pairs.ep, eps))[0]
        by_pos = defaultdict(list)
        for i in idx_b:
            by_pos[int(obs_pos[pairs.t[i]])].append((int(i), row_of_ep[int(pairs.ep[i])]))
        tokens = torch.from_numpy(data.tokens[eps].astype(np.int64)).to(device)
        out_idx, out = [], {s: [] for s in specs}

        def fn(pos, payload):
            items = by_pos[pos]
            rows = torch.as_tensor([r for _, r in items], device=device)
            out_idx.append(np.array([i for i, _ in items], dtype=np.int64))
            for s in specs:
                out[s].append(extract(payload, s[0], s[1])[rows].float().cpu().numpy())

        rec.run(tokens, sorted(by_pos), fn)
        yield np.concatenate(out_idx), {s: np.concatenate(out[s]) for s in specs}


def collect_many(it, n, dims, dtype=np.float32, memmap_dir=None, memmap_specs=(), memmap_dtype=np.float16):
    """Materialise streamed features into arrays [n, F] per spec: RAM fp32 by default, or on-disk fp16
    .npy memmaps for the specs listed in memmap_specs (spec section 4.5: sigma_full)."""
    X = {}
    for s, F in dims.items():
        if s in memmap_specs:
            path = Path(memmap_dir) / f"{s[0]}_L{s[1]}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            X[s] = np.lib.format.open_memmap(path, mode="w+", dtype=memmap_dtype, shape=(n, F))
        else:
            X[s] = np.empty((n, F), dtype=dtype)
    for idx, feats in it:
        for s in dims:
            X[s][idx] = feats[s].astype(X[s].dtype)
    for s in memmap_specs:
        X[s].flush()
    return X
