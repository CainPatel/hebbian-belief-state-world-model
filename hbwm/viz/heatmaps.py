"""Belief heatmap frames: 'watch the model's belief about the key decay' (spec section 4.6)."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hbwm.bdh.core import HBWMCore
from hbwm.device import select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.atlas import load_atlas
from hbwm.instrument.belief import belief_map
from hbwm.instrument.recorder import SigmaRecorder
from hbwm.train import load_checkpoint


@torch.no_grad()
def render_frames(model: HBWMCore, atlas, data: EpisodeData, episode: int, level: int, out_dir, device=None, max_steps=None):
    device = device if device is not None else next(model.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G, L = data.G, data.L
    n_steps = L + 1 if max_steps is None else min(L + 1, max_steps + 1)
    obs_pos = tk.obs_positions(L)[:n_steps]
    pos_to_t = {int(p): t for t, p in enumerate(obs_pos)}
    tokens = torch.from_numpy(data.tokens[episode : episode + 1].astype(np.int64)).to(device)
    objs = data.obj_type[episode]
    frames = []

    def fn(pos, payload):
        t = pos_to_t[pos]
        sigma = payload["sigma"][level, 0]  # nh,N,D
        fig, axes = plt.subplots(1, len(objs), figsize=(3.2 * len(objs), 3.4), squeeze=False)
        ax_, ay_ = data.agent_pos[episode, t]
        for j, (ax, k) in enumerate(zip(axes[0], objs)):
            bm = belief_map(model, sigma, level, atlas, int(k), G)
            ax.imshow(bm, cmap="magma", origin="upper")
            ox, oy = data.obj_pos[episode, t, j]
            ax.scatter([ox], [oy], marker="s", s=120, facecolors="none", edgecolors="cyan", linewidths=2)
            ax.scatter([ax_], [ay_], marker="*", s=140, c="white", edgecolors="black")
            ax.add_patch(plt.Rectangle((ax_ - 1.5, ay_ - 1.5), 3, 3, fill=False, edgecolor="white", linestyle="--"))
            sss = int(data.steps_since_seen[episode, t, j])
            ax.set_title(f"OBJ_{int(k)}  t={t}  since seen={sss}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        p = out_dir / f"frame_{t:03d}.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        frames.append(p)

    SigmaRecorder(model).run(tokens, list(map(int, obs_pos)), fn)
    return frames


def render_episode(run_dir, data_dir, split="probe_test", episode=0, level=None, out_dir=None, device=None, gif=True):
    run_dir = Path(run_dir)
    dev = select_device(device)
    model, _, _ = load_checkpoint(run_dir / "ckpt.pt", dev)
    atlas = load_atlas(run_dir / "probes" / "atlas.json")
    if level is None:
        done = json.loads((run_dir / "probes" / "done.json").read_text())
        spec = done.get("best_full_spec")
        level = int(spec.split("_L")[-1]) if spec else model.hcfg.n_layer - 1
    data = EpisodeData(data_dir, split)
    out = Path(out_dir) if out_dir else run_dir / "viz" / f"ep{episode}_L{level}"
    frames = render_frames(model, atlas, data, episode, level, out, dev)
    if gif:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(out / "anim.gif", [imageio.imread(p) for p in frames], duration=0.25)
        except ImportError:
            print("imageio not installed; frames only (uv sync --extra viz)")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default="data/grid9")
    ap.add_argument("--split", default="probe_test")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    frames = render_episode(args.run_dir, args.data, args.split, args.episode, args.level, gif=not args.no_gif)
    print(f"{len(frames)} frames -> {frames[0].parent}")


if __name__ == "__main__":
    main()
