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

## Study 1 headline

Study 1 is complete. The primary hypothesis H1 was **not supported**, and the preregistered kill
criterion fired. This is a preregistered negative result. The decisions below were taken strictly
under the rules frozen at preregistration commit
`e674b1da138f905670dde5571e1a1890b134fe36`, before any headline run.

Best learning rates (`runs/study1/best_lr.json`): bdh_g100 3e-3, lstm 3e-3, rwkv 3e-3. Every
headline run and both γ arms use lr 0.003.

### Aggregated results

Pasted from `runs/study1/results/results.md`, produced by
`uv run python -m hbwm.probes.evaluate --root runs --exp study1 --data data/grid9`. Numbers are
verbatim; only the heading levels are demoted one step so the tables nest under this section.

#### Probe accuracy (test, best level per seed; mean ± std over 3 seeds)

| model | feature | acc | chance | ceiling | #features | n_train | levels |
|---|---|---|---|---|---|---|---|
| bdh_g100 | sigma_full | 0.101 ± 0.007 | 0.011 | 1.000 | 524288 | 24000 | [3, 3, 4] |
| bdh_g100 | sigma_rownorm | 0.172 ± 0.008 | 0.011 | 1.000 | 8192 | 61400 | [4, 3, 4] |
| bdh_g100 | x_sparse | 0.062 ± 0.009 | 0.011 | 1.000 | 8192 | 61400 | [5, 4, 5] |
| bdh_g100 | resid | 0.040 ± 0.005 | 0.011 | 1.000 | 64 | 61400 | [5, 3, 5] |
| bdh_g099 | sigma_full | 0.099 ± 0.004 | 0.011 | 1.000 | 524288 | 24000 | [3, 3, 3] |
| bdh_g099 | sigma_rownorm | 0.180 ± 0.001 | 0.011 | 1.000 | 8192 | 61400 | [3, 3, 3] |
| bdh_g099 | x_sparse | 0.079 ± 0.002 | 0.011 | 1.000 | 8192 | 61400 | [3, 3, 4] |
| bdh_g099 | resid | 0.044 ± 0.004 | 0.011 | 1.000 | 64 | 61400 | [4, 5, 5] |
| bdh_g097 | sigma_full | 0.065 ± 0.001 | 0.011 | 1.000 | 524288 | 24000 | [3, 3, 3] |
| bdh_g097 | sigma_rownorm | 0.130 ± 0.002 | 0.011 | 1.000 | 8192 | 61400 | [3, 3, 3] |
| bdh_g097 | x_sparse | 0.068 ± 0.009 | 0.011 | 1.000 | 8192 | 61400 | [4, 4, 4] |
| bdh_g097 | resid | 0.038 ± 0.005 | 0.011 | 1.000 | 64 | 61400 | [5, 5, 3] |
| lstm | state_vec | 0.171 ± 0.006 | 0.011 | 1.000 | 1400 | 61400 | [None, None, None] |
| rwkv | state_vec | 0.218 ± 0.007 | 0.011 | 1.000 | 3520 | 61400 | [None, None, None] |

#### H1 — supported: **False** (margin 0.05)

| comparator | mean diff | paired diffs | passes |
|---|---|---|---|
| x_sparse | +0.039 | [0.044, 0.038, 0.034] | False |
| lstm | -0.070 | [-0.073, -0.07, -0.065] | False |
| rwkv | -0.117 | [-0.104, -0.132, -0.115] | False |

#### H2 — decay curves (accuracy by steps-since-seen bucket)

| model | 1-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ | graceful |
|---|---|---|---|---|---|---|---|
| bdh_g100 | 0.097 | 0.116 | 0.105 | 0.084 | 0.082 | 0.101 | True |
| bdh_g099 | 0.133 | 0.132 | 0.088 | 0.032 | 0.016 | 0.020 | False |
| bdh_g097 | 0.101 | 0.090 | 0.039 | 0.015 | 0.013 | 0.018 | False |
| lstm | 0.279 | 0.198 | 0.128 | 0.057 | 0.022 | 0.010 | False |
| rwkv | 0.322 | 0.271 | 0.184 | 0.080 | 0.029 | 0.010 | False |

Test pairs per bucket (probe_test, shared by all seeds):

| model | n(1-4) | n(5-8) | n(9-16) | n(17-32) | n(33-64) | n(65+) |
|---|---|---|---|---|---|---|
| bdh_g100 | 11660 | 10985 | 9204 | 5716 | 2812 | 662 |
| bdh_g099 | 11660 | 10985 | 9204 | 5716 | 2812 | 662 |
| bdh_g097 | 11660 | 10985 | 9204 | 5716 | 2812 | 662 |
| lstm | 11660 | 10985 | 9204 | 5716 | 2812 | 662 |
| rwkv | 11660 | 10985 | 9204 | 5716 | 2812 | 662 |

#### H3 — belief revision latency

| model | mean frac(latency ≤ 5) | supported | frac(≤5), not-visible steps only (exploratory) |
|---|---|---|---|
| bdh_g100 | 0.157 | False | 0.130 |
| bdh_g099 | 0.300 | False | 0.231 |
| bdh_g097 | 0.352 | False | 0.277 |
| lstm | 0.940 | True | 0.838 |
| rwkv | 0.953 | True | 0.845 |

#### H4 — sparsity (k90 = min top-k features reaching 90% of full accuracy)

| model | median k90 | #features | strong (≤256) | weak (≤1%) |
|---|---|---|---|---|
| bdh_g100 | 524288 | 524288 | False | False |
| bdh_g099 | 524288 | 524288 | False | False |
| bdh_g097 | 524288 | 524288 | False | False |
| lstm | 256 | 1400 | True | False |
| rwkv | 256 | 3520 | True | False |

#### Prediction quality

| model | params | lr | val CE | test CE | test CE (window) |
|---|---|---|---|---|---|
| bdh_g100 | 1577216 | 0.003 | 0.0242 | 0.0246 | 0.0250 |
| bdh_g099 | 1577216 | 0.003 | 0.0244 | 0.0249 | 0.0253 |
| bdh_g097 | 1577216 | 0.003 | 0.0268 | 0.0274 | 0.0284 |
| lstm | 1579310 | 0.003 | 0.0284 | 0.0291 | 0.0305 |
| rwkv | 1631168 | 0.003 | 0.0238 | 0.0242 | 0.0246 |

### Decisions under the preregistered rules

- **H1: not supported.** Mean σ_full accuracy is 0.101, which beats x_sparse (0.062) by only
  +0.039 and so misses the required 5-point margin, and loses outright to both the LSTM state
  (0.171, mean diff -0.070) and the RWKV state (0.218, mean diff -0.117), with every paired-by-seed
  difference against both baselines negative.
- **Kill criterion: fired.** The preregistered kill condition is "H1 fails against the LSTM state",
  which is exactly what happened, so the preregistered response applies: write the result up and
  stop or pivot rather than continue tuning Study 1.
- **H2: γ = 1.0 passes, everything else fails.** The γ = 1.0 arm satisfies the graceful test with
  acc(33-64)/acc(1-4) = 0.85 (0.846) and no bucket below half of its predecessor, while γ = 0.99
  (0.12), γ = 0.97 (0.13), the LSTM (0.08) and RWKV (0.09) all fall below the 0.5 ratio bar.
- **H3: not supported for any BDH arm.** The mean fraction of moved-and-re-observed episodes whose
  belief flips within 5 steps is 0.157 (γ = 1.0), 0.300 (γ = 0.99) and 0.352 (γ = 0.97), all below
  the preregistered 0.7 bar, while both baselines pass it (LSTM 0.940, RWKV 0.953).
- **H4: not supported for BDH.** Median k90 for every BDH arm is 524,288, the full feature count, so
  neither the strong criterion (median k90 ≤ 256) nor the weak one (≤ 1% of features) is met,
  whereas both baselines are strong-sparse at median k90 = 256.

### Required caveats

- **γ applies per token, not per environment step.** Each environment step is 12 tokens
  (`STEP_LEN = 12` in `hbwm/envs/tokenizer.py`: one action token plus an 11-token observation), so
  the effective per-step decay is γ^12: 0.99 becomes 0.886 per step (half-life about 5.7 steps) and
  0.97 becomes 0.694 per step (half-life about 1.9 steps). The γ arms are therefore far more
  aggressive than their names suggest, and the H2 γ-arm curves should be read against those
  per-step figures rather than against 0.99 and 0.97.
- **k90 = n_features means no proper sparse subset reached 90% of full accuracy.** The terminal grid
  point k = all comes from the spec and was adopted by a pre-analysis ruling; the strong and weak
  verdicts are unaffected by it.
- **The H3 headline rule counts the re-observation step itself**, at which the object is visible in
  the agent's 3x3 window. The exploratory not-visible-only variant is reported alongside it (BDH
  arms 0.130, 0.231, 0.277; baselines 0.838 and 0.845) and does not change any verdict.
- **The oracle-memory ceiling is 1.0 by construction.** Probe eligibility excludes stale pairs and
  static objects never move, so the ceiling is reported for completeness and is not an informative
  baseline.
- **σ_full probes train on 24,000 stratified examples against 61,400 for every comparator.** This is
  the preregistered protocol; the `n_train` column in the probe accuracy table makes the asymmetry
  visible.
- **Wall-clock and engineering notes.** Runs alternated between lid-closed dark-wake throttling and
  awake operation, which is why per-run wall-clock varies by roughly 2x at identical settings. One
  probe run was OOM-killed and re-run after memory fixes. A `batch_eps` change (64 to 32) made two
  cached fp16 probability archives differ by 1 ulp with zero prediction flips; this is an
  engineering note only, and no reported metric changed.

### Post-hoc, not preregistered

The four observations below are hypotheses for follow-up, not conclusions. None of them were
registered in advance and none of them change a decision above.

- **(a) σ row-norms beat σ_full.** The 8,192-dimensional row-norm view decodes at 0.172, above
  σ_full's 0.101 and level with the LSTM state's 0.171. One candidate explanation is that the
  524,288-dimensional linear probe trained on 24,000 examples underfits, so the σ_full number may
  understate what σ contains. This is untested.
- **(b) Prediction quality is not the bottleneck.** BDH's test CE is 0.0246 at γ = 1.0 against
  0.0291 for the LSTM and 0.0242 for RWKV, so BDH predicts as well as or better than the baselines
  it loses to on every probe hypothesis. The belief information demonstrably drives next-token
  behavior; what the probes show is that it is not linearly readable from σ at this scale and under
  this protocol.
- **(c) The γ = 1.0 H2 curve is graceful but uniformly low**, roughly 0.08 to 0.12 across all six
  buckets. That is the flatness of a weak signal rather than evidence of robust memory, and the
  graceful verdict should be read with that in mind.
- **(d) Every decoder sits far above chance and far below ceiling.** All accuracies are well above
  chance (0.011) and well below the 1.0 oracle ceiling, so exact-cell readout of out-of-view objects
  is hard for every architecture tested here, not only for BDH.

### What we would change (post-hoc)

None of the following was preregistered, and none of it is a claim that the result would change. As
a follow-up we would scale probe capacity and epochs to the feature count rather than holding them
fixed, and add MLP probes as a non-preregistered comparison, so that a null on σ_full separates
"not encoded" from "not linearly decodable at this probe budget". We would apply decay per
environment step rather than per token, so a named γ means what it appears to mean. We would raise
`n_train` for σ_full toward the 61,400 used by every other feature set, removing the training-set
asymmetry. We would define H3 from the first step at which the object is no longer visible, so the
headline number does not include a step where the answer is in the window. And we would try
readouts on σ deltas, or on the synapse view σ·encoder_v, rather than on raw σ entries.

### Exploratory belief heatmaps

Rendered with `uv run python -m hbwm.viz.heatmaps --run-dir runs/study1/bdh_g100_lr0.003/seed0
--episode {0,1}` at level L3 (the best σ_full level for that checkpoint), 97 frames plus `anim.gif`
per episode, written to `runs/study1/bdh_g100_lr0.003/seed0/viz/ep0_L3/` and `.../viz/ep1_L3/`.
Episode 1 is the first `probe_test` episode that is both moved and re-observed
(`reobserved_t = 34`); episode 0 is moved but never re-observed. These figures are exploratory and
were declared as such at preregistration.

- Episode 0, t = 48 (`docs/figures/belief_ep0_t48.png`): all three object maps are high-frequency
  and unstructured, with bright and dark cells scattered across the grid and no peak at the cyan
  true-object marker, including for OBJ_0, which is visible in the window at that step.
- Episode 1, t = 34, the re-observation step (`docs/figures/belief_ep1_reobs_t34.png`): OBJ_3 has
  just been seen (steps since seen = 0) inside the top-left window, yet its map is darkest exactly
  there and brightest along the right edge, and the three object panels are near-copies of one
  another.
- Episode 1, t = 0 (`runs/study1/bdh_g100_lr0.003/seed0/viz/ep1_L3/frame_000.png`): the three
  panels are nearly identical to each other, a smooth gradient that is dark near the agent at
  top-left and bright toward the bottom-right, again with no object-specific structure.

Read honestly, these maps look dominated by an object-independent component and show no localized
peak at the true cell, which is consistent with the H1, H3 and H4 results rather than in tension
with them.

### Probe wall-clock

`elapsed_s` from the 15 `runs/study1/*/seed*/probes/done.json` files:

| model | seed 0 | seed 1 | seed 2 | total (s) |
|---|---|---|---|---|
| bdh_g100 | 5460.4 | 5313.2 | 5131.4 | 15905.0 |
| bdh_g099 | 5661.5 | 5612.5 | 5611.1 | 16885.1 |
| bdh_g097 | 5558.9 | 5442.7 | 5371.2 | 16372.8 |
| lstm | 76.7 | 75.0 | 74.2 | 225.9 |
| rwkv | 279.5 | 280.0 | 279.9 | 839.4 |

All 15 checkpoints total 50,228.2 s (about 13.95 h). The BDH checkpoints cost 5,131 to 5,662 s each
because each one runs 21 probe specs including three σ_full levels at 524,288 features; the
baselines run a single `state_vec` spec and cost 74 to 280 s. `best_full_spec` is `sigma_full_L3`
for 8 of the 9 BDH checkpoints and `sigma_full_L4` for `bdh_g100_lr0.003/seed2`.

### Figures and provenance

- `docs/figures/h2_curves.png` and `docs/figures/h4_curves.png` (copies of
  `runs/study1/results/h2_curves.png` and `runs/study1/results/h4_curves.png`).
- `docs/figures/belief_ep0_t48.png` and `docs/figures/belief_ep1_reobs_t34.png` (copies of
  `runs/study1/bdh_g100_lr0.003/seed0/viz/ep0_L3/frame_048.png` and `.../viz/ep1_L3/frame_034.png`).
- Full frame sets and animations: `runs/study1/bdh_g100_lr0.003/seed0/viz/ep0_L3/` and
  `runs/study1/bdh_g100_lr0.003/seed0/viz/ep1_L3/` (gitignored, local only).
- Aggregation source: `runs/study1/results/{results.md,h1.json,h2.json,h3.json,h4.json,perplexity.json,table.json}`.
- Preregistration commit: `e674b1da138f905670dde5571e1a1890b134fe36`. All H1 to H4 decisions above
  were made by the rules frozen in that commit, with no post-hoc adjustment to any threshold.

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

## E2/E3 — seeds and γ arms

Source: `runs/study1/{bdh_g100,lstm,rwkv,bdh_g099,bdh_g097}_lr0.003/seed{0,1,2}/final.json`.
All twelve headline runs use lr 0.003: `runs/study1/best_lr.json` selected 3e-3 for all three
models (`{"bdh_g100": 0.003, "lstm": 0.003, "rwkv": 0.003}`), and the γ arms inherit the primary
BDH learning rate by the preregistered protocol. `bdh_g100_lr0.003/seed0` is shared with the E1
sweep and is not re-run.

| phase | model / γ arm | seed 0 | seed 1 | seed 2 | mean best val CE |
|---|---|---|---|---|---|
| E2 | bdh_g100 (γ = 1.00) | 0.0240 | 0.0252 | 0.0232 | 0.0242 |
| E2 | lstm | 0.0286 | 0.0275 | 0.0290 | 0.0284 |
| E2 | rwkv | 0.0235 | 0.0243 | 0.0237 | 0.0238 |
| E3 | bdh_g099 (γ = 0.99) | 0.0242 | 0.0244 | 0.0246 | 0.0244 |
| E3 | bdh_g097 (γ = 0.97) | 0.0268 | 0.0272 | 0.0264 | 0.0268 |

All 15 runs reached step 4000 with a finite `best_val_ce`; `best_step` is 4000 everywhere except
`bdh_g100_lr0.003/seed1` and `bdh_g097_lr0.003/seed0`, whose best eval landed at step 3800. Spread
across seeds is small for every arm (max within-arm range 0.0020 nats, for bdh_g100). Parameter
counts are unchanged from E1: 1,577,216 for every BDH arm, 1,579,310 for the LSTM, 1,631,168 for
RWKV.

**Wall-clock:** BDH arms ran at 0.117–0.256 steps/s (15,635–34,269 s per seed), LSTM at
5.67–5.72 steps/s (about 700 s per seed), RWKV at 0.465–0.501 steps/s (7,981–8,606 s per seed).
The outlier is `bdh_g099_lr0.003/seed2` at 34,268.5 s (0.117 steps/s), roughly twice the wall-clock
of its sibling seeds; this is the lid-closed dark-wake throttling described under E1, not a
difference in the run itself. Its `best_val_ce` (0.0246) is in line with seeds 0 and 1.
