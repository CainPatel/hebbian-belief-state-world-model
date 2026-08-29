"""EXPLORATORY, not preregistered. Data-only: no model, no GPU.

Question: did the next-token objective ever reward remembering the ABSOLUTE cell of an
out-of-view object? The model predicts each observation, which is X, Y and 9 window cells.
Knowing where an unseen object is can only reduce loss at the moment that object is inside
the 3x3 window. So the value of memory is bounded by how much of the predicted content
consists of objects returning to view after an absence, and by whether they are still where
memory says they are.
"""

import numpy as np

from hbwm.envs import tokenizer as tk

D = "/Users/cainpatel/Coding/Claude Code/Hebbian Belief State World Model/.claude/worktrees/study1-impl/data/grid9"
z = np.load(f"{D}/probe_test.npz")
vis = z["visible"]                    # [n, T, k] object visible at step t
obj_pos = z["obj_pos"]                # [n, T, k, 2]
moved, move_obj = z["moved"], z["move_obj"]
n, T, K = vis.shape

# ---- 1. how much of what the model predicts is object content at all ----
tokens = z["tokens"].astype(np.int64)
obs_pos = tk.obs_positions(96)                       # last token index of each observation
# an observation is X, Y then 9 window cells, ending at obs_pos[t]
win_idx = (obs_pos[:, None] - 8 + np.arange(9)).reshape(-1)
win = tokens[:, win_idx]
is_obj = win >= tk.OBJ_BASE
print(f"observations per episode      : {len(obs_pos)}")
print(f"window cells predicted/episode: {win.shape[1]}  (9 per observation)")
print(f"window cells that are objects : {is_obj.mean():.4f}")
print(f"                        walls : {(win == tk.WALL_TOK).mean():.4f}")
print(f"                        empty : {(win == tk.EMPTY_TOK).mean():.4f}")

# ---- 2. of every step where an object is visible, is this a RETURN after absence? ----
seen_before = np.cumsum(vis, axis=1) - vis > 0        # visible at some earlier step
prev_vis = np.zeros_like(vis)
prev_vis[:, 1:] = vis[:, :-1]
first_sight = vis & ~seen_before
returning = vis & seen_before & ~prev_vis             # came back into view this step
continuing = vis & prev_vis
tot_vis = vis.sum()
print(f"\nobject-visible (ep,t,k) events: {tot_vis}")
print(f"  first sighting               : {first_sight.sum()/tot_vis:.4f}")
print(f"  returning after an absence   : {returning.sum()/tot_vis:.4f}")
print(f"  still in view from last step : {continuing.sum()/tot_vis:.4f}")

# ---- 3. when an object returns, is it where memory would say? ----
cells = obj_pos[..., 1] * 9 + obj_pos[..., 0]         # [n, T, k] cell id
t_idx = np.broadcast_to(np.arange(T)[None, :, None], vis.shape)
last_t = np.maximum.accumulate(np.where(prev_vis, t_idx, -1), axis=1)
last_cell = np.take_along_axis(cells, np.clip(last_t, 0, None), axis=1)
r = returning & (last_t >= 0)
same = (last_cell == cells)[r]
print(f"\nreturns where the object is STILL where last seen: {same.mean():.4f}")
print(f"returns where it MOVED while out of view          : {1 - same.mean():.4f}")

# ---- 4. the bound: share of predicted window cells memory could ever help with ----
# a returning object occupies exactly one window cell at that step
help_cells = (returning & (last_cell == cells)).sum()
total_cells = win.size
print("\nwindow cells where memory of an absent object could help:")
print(f"  {help_cells} of {total_cells}  =  {help_cells/total_cells:.5f}")
print(f"  as a share of all predicted tokens ({tokens[:, 1:].size}): "
      f"{help_cells/tokens[:, 1:].size:.5f}")

# ---- 5. how long are the absences the probe is scored on? ----
# steps_since_seen is 0 at the return step itself (the object is visible again), so the
# absence is t minus the last step it was actually visible.
gap = (t_idx - last_t)[r]
print(f"\nabsence length at the moment of return: median {np.median(gap):.0f}, "
      f"p75 {np.quantile(gap, 0.75):.0f}, p90 {np.quantile(gap, 0.9):.0f}, max {gap.max()}")
for b in (1, 2, 4, 8, 16, 32):
    print(f"  returns after an absence of <= {b:2d} steps: {(gap <= b).mean():.4f}")

# Study 2 scored the probe on absences of >= 1 step, bucketed out to 65+. How much of the
# objective's own object content sits in those long-absence buckets?
long_ret = (returning & (last_cell == cells) & ((t_idx - last_t) >= 9)).sum()
print(f"\nwindow cells from a return after >= 9 steps away: {long_ret} of {total_cells} "
      f"= {long_ret/total_cells:.6f}")
