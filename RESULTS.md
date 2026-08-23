# HBWM Results Log

All numbers are from `runs/`; negative results are recorded too.

## E0 — Shakespeare sanity (core, γ = 1, D=128, 6 levels)

`uv run python -m hbwm.sanity_shakespeare --steps 1000`

| device | params | step 0 val loss | step 1000 val loss | wall-clock (s) |
|---|---|---|---|---|
| mps | 6,356,992 | 5.600 | 1.603 | 5176.8 |

Val loss fell monotonically from ln(256) ≈ 5.5 nats at step 0 to well below the ≈2.2 nats bar by
step 100 (2.477), continuing down to a best of **1.474 at step 600**; steps 600-1000 wobble in the
1.47-1.60 range (700: 1.542, 800: 1.604, 900: 1.501, 1000: 1.603) rather than strictly decreasing,
which is within the "minor wobbles allowed" tolerance in the task brief. Full per-100-step curve in
`runs/e0_shakespeare/log.json`.

**Timing caveat:** this run's wall-clock is not representative of awake MPS throughput. The machine
was asleep/throttled for part of the run (closed lid, dark-wake only) — the first 100 steps include
about 40 minutes of sleep, and steps 100-1000 ran at roughly 2.5-3.75 s/step under dark-wake
throttling. A clean foreground benchmark of the same config measured about 2.1 s/step after a
one-time MPS warm-up. Use the foreground figure, not the 5176.8 s elapsed_s above, to calibrate
later run-time estimates.

Sample (top-k=3), first two lines:
```
To be or any thing,
And the dishonour of the fair worldly knaves.
```

## Dataset grid9

Generated with `uv run python -m hbwm.envs.dataset --config experiments/data/grid9.json` (wall-clock ~31s for 27,000 episodes; grid=9x9, 3 objects, episode_len=96, T=1164 tokens/episode).

Sanity check (`probe_test` split, n=2000):
```
uv run python -c "from hbwm.envs.dataset import EpisodeData as E; d=E('data/grid9','probe_test'); import numpy as np; print(d.n, d.T, d.moved.mean(), (d.reobserved_t>=0).mean(), d.visible.mean())"
2000 1164 0.5205 0.311 0.09749828178694159
```

- Fraction of episodes with an object moved: **0.5205** (expected ~0.45-0.5)
- Fraction of episodes where the moved object is re-observed before the episode ends: **0.311** (expected ~0.3-0.45)
- Fraction of (t, obj) pairs where the object is visible: **0.0975** (expected ~0.1-0.2)

## Calibration

`uv run python -m hbwm.train --config experiments/train/bdh_g100.json --seed 0 --max-steps 300 --out-root runs_calib` (γ = 1.0 preregistered config, real `data/grid9` data, MPS). No `PYTORCH_ENABLE_MPS_FALLBACK` was needed — the run completed cleanly on plain MPS.

| n_params | device | val_ce @ step 0 | val_ce @ step 300 | seconds (300 steps) | steps_per_sec |
|---|---|---|---|---|---|
| 1,577,216 | mps | 3.7228 | 0.1953 | 1115.1 | 0.269 |

`val_ce` at step 0 (3.7228) is close to the ln(34) ≈ 3.526 uniform-prior bound and fell monotonically to 0.1953 by step 300 (well under the < 1.0 bar), confirming the observation tokens are highly predictable as expected. Full per-eval and per-20-step curve in `runs_calib/study1/bdh_g100_lr0.001/seed0/metrics.jsonl`.

Projected 4000-step wall-clock at this measured rate: 4000 / 0.269 ≈ **14,870 s (≈ 4.13 h)** per seed (equivalently 1115.1 s × 4000/300 ≈ 14,868 s).

**Throttling caveat:** this calibration was measured while the machine was in a reduced-power "dark wake" state (lid closed, GPU running at roughly 50-70% of normal speed), plus the first step of the fresh process pays a one-time multi-minute MPS warm-up. Awake, un-throttled MPS throughput is expected to be materially higher than 0.269 steps/sec, so the projected 4000-step figure above is a conservative upper bound, not a clean-hardware benchmark.
