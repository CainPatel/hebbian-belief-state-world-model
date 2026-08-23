import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from hbwm.config import load_config, to_dict
from hbwm.envs import tokenizer as tk
from hbwm.envs.episode import run_episode, stale_mask, steps_since_seen
from hbwm.envs.gridworld import GridConfig

SPLIT_ORDER = ["model_train", "model_val", "probe_train", "probe_val", "probe_test"]


def _default_splits():
    return {"model_train": 20000, "model_val": 1000, "probe_train": 3000, "probe_val": 1000, "probe_test": 2000}


@dataclasses.dataclass
class DataConfig:
    grid: GridConfig = dataclasses.field(default_factory=GridConfig)
    splits: dict = dataclasses.field(default_factory=_default_splits)
    seed_base: int = 0
    out_dir: str = "data/grid9"


def split_seed_ranges(cfg: DataConfig) -> dict[str, tuple[int, int]]:
    start = cfg.seed_base
    ranges = {}
    for name in SPLIT_ORDER:
        if name in cfg.splits:
            n = int(cfg.splits[name])
            ranges[name] = (start, start + n)
            start += n
    return ranges


def build_split(cfg: DataConfig, name: str) -> dict[str, np.ndarray]:
    s0, s1 = split_seed_ranges(cfg)[name]
    eps = [run_episode(cfg.grid, seed) for seed in range(s0, s1)]
    L = cfg.grid.episode_len
    out = {
        "tokens": np.stack([tk.encode_episode(e.actions, e.agent_pos, e.windows) for e in eps]).astype(np.int16),
        "loss_mask": tk.loss_mask(L),
        "window_mask": tk.window_mask(L),
        "agent_pos": np.stack([e.agent_pos for e in eps]),
        "obj_pos": np.stack([e.obj_pos for e in eps]),
        "obj_type": np.stack([e.obj_type for e in eps]),
        "visible": np.stack([e.visible for e in eps]),
        "steps_since_seen": np.stack([steps_since_seen(e.visible) for e in eps]),
        "stale": np.stack([stale_mask(e) for e in eps]),
        "moved": np.array([e.moved for e in eps]),
        "move_t": np.array([e.move_t for e in eps]),
        "move_obj": np.array([e.move_obj for e in eps]),
        "move_from": np.stack([e.move_from for e in eps]),
        "move_to": np.stack([e.move_to for e in eps]),
        "reobserved_t": np.array([e.reobserved_t for e in eps]),
        "policy": np.array([e.policy for e in eps]),
        "seed": np.array([e.seed for e in eps]),
    }
    return out


def generate(cfg: DataConfig, only: list[str] | None = None) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_ORDER:
        if name not in cfg.splits or (only and name not in only):
            continue
        arrays = build_split(cfg, name)
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        print(f"wrote {name}: {arrays['tokens'].shape}")
    meta = {
        "T": tk.seq_len(cfg.grid.episode_len),
        "vocab_size": tk.VOCAB_SIZE,
        "grid": to_dict(cfg.grid),
        "splits": cfg.splits,
        "seed_ranges": split_seed_ranges(cfg),
        "seed_base": cfg.seed_base,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


class EpisodeData:
    """One split, fully in memory (int16 tokens + ground-truth arrays)."""

    def __init__(self, data_dir: str, split: str):
        z = np.load(Path(data_dir) / f"{split}.npz")
        for k in z.files:
            setattr(self, k, z[k])
        meta = json.loads((Path(data_dir) / "meta.json").read_text())
        self.G = int(meta["grid"]["size"])
        self.L = int(meta["grid"]["episode_len"])
        self.n, self.T = self.tokens.shape
        self.n_obj = self.obj_pos.shape[2]
        self._mask_t = torch.from_numpy(self.loss_mask[1:].copy())

    def batch_at(self, indices, device):
        tok = torch.from_numpy(self.tokens[np.asarray(indices)].astype(np.int64))
        return tok[:, :-1].to(device), tok[:, 1:].to(device), self._mask_t.to(device)

    def get_batch(self, rng: np.random.Generator, batch_size: int, device):
        return self.batch_at(rng.integers(0, self.n, size=batch_size), device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    generate(load_config(args.config, DataConfig), args.only)


if __name__ == "__main__":
    main()
