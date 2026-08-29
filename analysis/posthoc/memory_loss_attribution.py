"""EXPLORATORY, not preregistered.

Data-only analysis showed that window cells needing memory of an out-of-view object are
0.50% of predicted tokens. That is a token COUNT, not a loss SHARE. Test CE is 0.0246 nats
because most tokens (empty cells, walls, the agent's own coordinates) are near-deterministic.
This measures what share of each model's actual residual loss sits on those cells.

One forward pass per checkpoint over probe_test. No training, no probes.
"""

import numpy as np
import torch
import torch.nn.functional as F

from hbwm.device import select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.train import load_checkpoint

W = "/Users/cainpatel/Coding/Claude Code/Hebbian Belief State World Model/.claude/worktrees/study1-impl"
DATA = f"{W}/data/grid9"
d = EpisodeData(DATA, "probe_test")
n, T = d.tokens.shape
obs_pos = tk.obs_positions(d.L)

# ---- label every window-cell token position ----
vis, obj_pos, agent_pos = d.visible, d.obj_pos, d.agent_pos
cells = obj_pos[..., 1] * d.G + obj_pos[..., 0]
prev_vis = np.zeros_like(vis)
prev_vis[:, 1:] = vis[:, :-1]
seen_before = np.cumsum(vis, axis=1) - vis > 0
t_idx = np.broadcast_to(np.arange(vis.shape[1])[None, :, None], vis.shape)
last_t = np.maximum.accumulate(np.where(prev_vis, t_idx, -1), axis=1)
last_cell = np.take_along_axis(cells, np.clip(last_t, 0, None), axis=1)
returning = vis & seen_before & ~prev_vis
mem_relevant = returning & (last_cell == cells) & (last_t >= 0)   # [n, T_obs, k]

# slot of each visible object in its 3x3 window: dy in (-1,0,1) outer, dx inner
dx = obj_pos[..., 0] - agent_pos[..., 0:1]
dy = obj_pos[..., 1] - agent_pos[..., 1:2]
slot = (dy + 1) * 3 + (dx + 1)

# full-token index of window slot s at observation t is obs_pos[t] - 8 + s
label = np.zeros((n, T), dtype=np.int8)          # 0 = other
win_full = (obs_pos[:, None] - 8 + np.arange(9))  # [T_obs, 9]
tokens = d.tokens.astype(np.int64)
label[:, win_full.reshape(-1)] = 1                # 1 = window cell, not memory-relevant
ep_i, t_i, k_i = np.nonzero(mem_relevant)
label[ep_i, win_full[t_i, slot[ep_i, t_i, k_i]]] = 2   # 2 = memory-relevant returning object
# 3 = an object cell that is NOT memory relevant (continuing, first sight, or it moved)
obj_mask = np.zeros((n, T), dtype=bool)
ep_v, t_v, k_v = np.nonzero(vis)
obj_mask[ep_v, win_full[t_v, slot[ep_v, t_v, k_v]]] = True
label[(label == 1) & obj_mask] = 3

# shift to target frame: target index i predicts full index i+1
lab_t = label[:, 1:]
mask_t = d.loss_mask[1:]

NAMES = {0: "non-window tokens", 1: "window: empty/wall", 3: "window: object, no memory needed",
         2: "window: object returning, MEMORY RELEVANT"}

rows = {}
for stem in ("bdh_g100", "lstm", "rwkv"):
  for seed in (0, 1, 2):
    dev = select_device(None)
    model, _, _ = load_checkpoint(f"{W}/runs/study1/{stem}_lr0.003/seed{seed}/ckpt.pt", dev)
    model.eval()
    tot = np.zeros(4)
    cnt = np.zeros(4)
    with torch.no_grad():
        for b0 in range(0, n, 100):
            tokb = torch.from_numpy(tokens[b0 : b0 + 100]).to(dev)
            logits, _ = model(tokb[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 tokb[:, 1:].reshape(-1), reduction="none")
            ce = ce.view(tokb.size(0), -1).float().cpu().numpy()
            lb = lab_t[b0 : b0 + 100]
            for c in (0, 1, 2, 3):
                m = (lb == c) & mask_t[None, :]
                tot[c] += ce[m].sum()
                cnt[c] += m.sum()
    grand = tot.sum()
    rows.setdefault(stem, []).append((grand / cnt.sum(), tot[2] / cnt[2], tot[2] / grand,
                                      tot[3] / cnt[3]))
    print(f"{stem} seed{seed}: overall {grand/cnt.sum():.5f} | "
          f"MEMORY-RELEVANT mean CE {tot[2]/cnt[2]:.4f} share {tot[2]/grand:.4f} | "
          f"other-object mean CE {tot[3]/cnt[3]:.4f}")

print("\n=== mean over 3 seeds ===")
print(f"{'model':10s} {'overall CE':>11s} {'mem-cell CE':>12s} {'mem loss share':>15s} "
      f"{'other-obj CE':>13s}")
for stem, v in rows.items():
    a = np.array(v).mean(axis=0)
    print(f"{stem:10s} {a[0]:11.5f} {a[1]:12.4f} {a[2]:15.4f} {a[3]:13.4f}")
