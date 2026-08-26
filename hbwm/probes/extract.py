from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hbwm.device import release_memory
from hbwm.envs import tokenizer as tk
from hbwm.instrument.features import extract
from hbwm.instrument.recorder import make_recorder

__all__ = ["collect_many", "iter_features", "release_memory"]

MEMMAP_FLUSH_EVERY = 8  # batches; caps the dirty pages the fp16 sigma_full memmaps accumulate
RELEASE_EVERY = 4  # batches between release_memory() calls; per-batch cost 2.5x on the CPU test suite


def _batch_features(rec, tokens_np, specs, by_pos, device):
    """One recorder pass over a batch of episodes -> (pair indices, {spec: X[n_i, F]}).

    A frame of its own on purpose: the device token tensor, the per-position buffers and the callback
    that closes over them all die when this returns, so the caller's periodic release_memory() finds
    them unreferenced instead of pinned by a still-live loop body.
    """
    tokens = torch.from_numpy(tokens_np.astype(np.int64)).to(device)
    out_idx, out = [], {s: [] for s in specs}

    def fn(pos, payload):
        items = by_pos[pos]
        rows = torch.as_tensor([r for _, r in items], device=device)
        out_idx.append(np.array([i for i, _ in items], dtype=np.int64))
        for s in specs:
            out[s].append(extract(payload, s[0], s[1])[rows].float().cpu().numpy())

    rec.run(tokens, sorted(by_pos), fn)
    return np.concatenate(out_idx), {s: np.concatenate(out[s]) for s in specs}


def iter_features(model, data, pairs, specs, batch_eps=64, device=None):
    """Yield (pair_indices, {spec: X[n_i, F]}) batches; each pair index is yielded exactly once."""
    device = device if device is not None else next(model.parameters()).device
    rec = make_recorder(model)
    obs_pos = tk.obs_positions(data.L)
    unique_eps = np.unique(pairs.ep)
    for b, b0 in enumerate(range(0, len(unique_eps), batch_eps), start=1):
        eps = unique_eps[b0 : b0 + batch_eps]
        row_of_ep = {int(e): i for i, e in enumerate(eps)}
        idx_b = np.where(np.isin(pairs.ep, eps))[0]
        by_pos = defaultdict(list)
        for i in idx_b:
            by_pos[int(obs_pos[pairs.t[i]])].append((int(i), row_of_ep[int(pairs.ep[i])]))
        yield _batch_features(rec, data.tokens[eps], specs, by_pos, device)
        # Study 1 makes ~6 full passes per checkpoint; without this the MPS allocator keeps every
        # pass's blocks cached. gc.collect() is not free, so amortise it over RELEASE_EVERY batches.
        if b % RELEASE_EVERY == 0:
            release_memory(device)


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
    for b, (idx, feats) in enumerate(it, start=1):
        for s in dims:
            X[s][idx] = feats[s].astype(X[s].dtype)
        if memmap_specs and b % MEMMAP_FLUSH_EVERY == 0:
            for s in memmap_specs:  # write dirty pages out as we go; the whole file is ~25 GB per level
                X[s].flush()
    for s in memmap_specs:
        X[s].flush()
    return X
