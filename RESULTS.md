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

### Baselines — LSTM / RWKV (param-matched, both devices)

Param-matching solve (`uv run python -m hbwm.baselines.matching`, target = bdh_g100's 1,577,216 params):
`lstm_hidden: 350` → 1,579,310 params (+0.13%); `rwkv_width: 176` → 1,631,168 params (+3.42%). Both
within the ±5% fairness-protocol tolerance; configs preregistered at `experiments/train/lstm.json`
and `experiments/train/rwkv.json` (same hyper-parameters as `bdh_g100.json`).

Each baseline was calibrated for 300 steps on both MPS and CPU (`--out-root runs_calib` for MPS,
`runs_calib_cpu` for CPU), machine in the same dark-wake throttled state as the BDH calibration above:

| model | device | n_params | val_ce @ step 0 | val_ce @ step 300 | seconds (300 steps) | steps_per_sec |
|---|---|---|---|---|---|---|
| lstm | mps | 1,579,310 | 3.5009 | 0.2964 | 49.0 | 6.123 |
| lstm | cpu | 1,579,310 | 3.5009 | 0.2994 | 616.1 | 0.487 |
| rwkv | mps | 1,631,168 | 3.8017 | 0.1172 | 672.2 | 0.446 |
| rwkv | cpu | 1,631,168 | 3.8017 | 0.1120 | 1831.1 | 0.164 |

Both baselines' `val_ce` fell from close to the ln(34) ≈ 3.526 uniform-prior bound at step 0 to well
under 1.0 by step 300 on every device, confirming both architectures learn on `data/grid9` as
expected regardless of device. Full per-eval curves in
`runs_calib{,_cpu}/study1/{lstm,rwkv}_lr0.001/seed0/metrics.jsonl`.

**Device decision:** MPS is faster than CPU for both baselines (LSTM: 6.123 vs 0.487 steps/s, a
~12.6x speedup; RWKV: 0.446 vs 0.164 steps/s, a ~2.7x speedup) — neither is a tie, so per the
higher-steps_per_sec rule both `experiments/train/lstm.json` and `experiments/train/rwkv.json` are
pinned to `"device": "mps"` for the Phase 5 matrix.

## Preregistration

Decision rules: see README.md. Preregistration commit: `e674b1da138f905670dde5571e1a1890b134fe36`.

## Study 1 headline (filled by Task 34)

- Best LRs: (from runs/study1/best_lr.json)
- Probe accuracy table, H1–H4 decisions, prediction-quality table: copied from runs/study1/results/results.md
- Figures: runs/study1/results/h2_curves.png, h4_curves.png; heatmap frames under runs/study1/<run>/viz/

## E1 — LR sweep (seed 0)

Source: `runs/study1/{bdh_g100,lstm,rwkv}_lr{0.0003,0.001,0.003}/seed0/final.json`.

| model | lr | best val CE | best step | wall-clock (s) | steps/sec |
|---|---|---|---|---|---|
| bdh_g100 | 0.0003 | 0.0329 | 4000 | 14631.1 | 0.2734 |
| bdh_g100 | 0.001 | 0.0252 | 4000 | 17432.1 | 0.2295 |
| bdh_g100 | 0.003 | 0.0240 | 4000 | 16402.8 | 0.2439 |
| lstm | 0.0003 | 0.0496 | 4000 | 703.8 | 5.6834 |
| lstm | 0.001 | 0.0339 | 4000 | 705.8 | 5.6673 |
| lstm | 0.003 | 0.0286 | 4000 | 706.0 | 5.6657 |
| rwkv | 0.0003 | 0.0315 | 4000 | 7994.1 | 0.5004 |
| rwkv | 0.001 | 0.0244 | 4000 | 8012.8 | 0.4992 |
| rwkv | 0.003 | 0.0235 | 4000 | 7981.1 | 0.5012 |

Selected best LR per model (lowest best_val_ce): **bdh_g100 → 0.003** (0.0240), **lstm → 0.003** (0.0286), **rwkv → 0.003** (0.0235).

**Wall-clock varies ~2× across runs:** the machine alternated between lid-closed dark-wake throttling (~0.23–0.27 steps/s BDH) and awake operation.

**E1 notes:** all 9 `final.json` files present; all `best_val_ce` values finite (range 0.0235–0.0496); no divergence observed.
