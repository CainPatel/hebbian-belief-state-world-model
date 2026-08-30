# HBWM Results Log

All numbers are from `runs/`; negative results are recorded too.

## E0: Shakespeare sanity (core, γ = 1, D=128, 6 levels)

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
was asleep/throttled for part of the run (closed lid, dark-wake only): the first 100 steps include
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

`uv run python -m hbwm.train --config experiments/train/bdh_g100.json --seed 0 --max-steps 300 --out-root runs_calib` (γ = 1.0 preregistered config, real `data/grid9` data, MPS). No `PYTORCH_ENABLE_MPS_FALLBACK` was needed; the run completed cleanly on plain MPS.

| n_params | device | val_ce @ step 0 | val_ce @ step 300 | seconds (300 steps) | steps_per_sec |
|---|---|---|---|---|---|
| 1,577,216 | mps | 3.7228 | 0.1953 | 1115.1 | 0.269 |

`val_ce` at step 0 (3.7228) is close to the ln(34) ≈ 3.526 uniform-prior bound and fell monotonically to 0.1953 by step 300 (well under the < 1.0 bar), confirming the observation tokens are highly predictable as expected. Full per-eval and per-20-step curve in `runs_calib/study1/bdh_g100_lr0.001/seed0/metrics.jsonl`.

Projected 4000-step wall-clock at this measured rate: 4000 / 0.269 ≈ **14,870 s (≈ 4.13 h)** per seed (equivalently 1115.1 s × 4000/300 ≈ 14,868 s).

**Throttling caveat:** this calibration was measured while the machine was in a reduced-power "dark wake" state (lid closed, GPU running at roughly 50-70% of normal speed), plus the first step of the fresh process pays a one-time multi-minute MPS warm-up. Awake, un-throttled MPS throughput is expected to be materially higher than 0.269 steps/sec, so the projected 4000-step figure above is a conservative upper bound, not a clean-hardware benchmark.

### Baselines: LSTM / RWKV (param-matched, both devices)

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
~12.6x speedup; RWKV: 0.446 vs 0.164 steps/s, a ~2.7x speedup); neither is a tie, so per the
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

#### H1, supported: **False** (margin 0.05)

| comparator | mean diff | paired diffs | passes |
|---|---|---|---|
| x_sparse | +0.039 | [0.044, 0.038, 0.034] | False |
| lstm | -0.070 | [-0.073, -0.07, -0.065] | False |
| rwkv | -0.117 | [-0.104, -0.132, -0.115] | False |

#### H2: decay curves (accuracy by steps-since-seen bucket)

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

#### H3: belief revision latency

| model | mean frac(latency ≤ 5) | supported | frac(≤5), not-visible steps only (exploratory) |
|---|---|---|---|
| bdh_g100 | 0.157 | False | 0.130 |
| bdh_g099 | 0.300 | False | 0.231 |
| bdh_g097 | 0.352 | False | 0.277 |
| lstm | 0.940 | True | 0.838 |
| rwkv | 0.953 | True | 0.845 |

#### H4: sparsity (k90 = min top-k features reaching 90% of full accuracy)

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

## E1: LR sweep (seed 0)

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

**Wall-clock varies ~2× across runs:** the machine alternated between lid-closed dark-wake throttling (~0.23 to 0.27 steps/s BDH) and awake operation.

**E1 notes:** all 9 `final.json` files present; all `best_val_ce` values finite (range 0.0235 to 0.0496); no divergence observed.

## E2/E3: seeds and γ arms

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

**Wall-clock:** BDH arms ran at 0.117 to 0.256 steps/s (15,635 to 34,269 s per seed), LSTM at
5.67 to 5.72 steps/s (about 700 s per seed), RWKV at 0.465 to 0.501 steps/s (7,981 to 8,606 s per seed).
The outlier is `bdh_g099_lr0.003/seed2` at 34,268.5 s (0.117 steps/s), roughly twice the wall-clock
of its sibling seeds; this is the lid-closed dark-wake throttling described under E1, not a
difference in the run itself. Its `best_val_ce` (0.0246) is in line with seeds 0 and 1.

---

## Post-hoc analyses (exploratory, not preregistered)

**Read this header as a fence.** Everything below was computed after Study 1 closed, on saved
artifacts only, with no model retraining and no new preregistration. **No preregistered decision
changes.** H1 remains not supported and the kill criterion remains fired; H2 remains a pass for
gamma = 1.0 only; H3 and H4 remain not supported for BDH. Nothing below reopens a threshold, and no
threshold below was chosen after seeing the data except where the text says so explicitly.

Scripts: `analysis/posthoc/{sigma_structure.py,spatial_locality.py,spatial_locality_buckets.py}`.
They are read-only on `runs/` and `data/`.

### (a) Sigma is structurally sparse and lowish rank

**Method.** One checkpoint: `runs/study1/bdh_g100_lr0.003/seed0/ckpt.pt` (gamma = 1.0, seed 0, step
4000, val CE 0.0240). The first 64 of the 2,000 `probe_test` episodes are driven through
`hbwm.instrument.recorder.SigmaRecorder` at `plasticity="full"`. Statistics are computed per
(episode, head) slice of sigma, which has shape `[N = 2048, D = 64]`, giving 64 x 4 = **256 samples
per cell**, reported as mean +/- std across that pooled population. SVDs are taken on CPU in double
precision. "final" is the last input position of the episode (index 1162); "early" is the last token
of observation o_5 (index 71). Write concentration accumulates the exact write increment
alpha * q (outer) x at all 1,163 input positions, reconstructed through `Attention.rope` from the
recorder's `x_sparse` and `resid` payloads, not by finite difference. Levels 0, 3 and 5 of the six
shared `HBWMCore` levels are compared; level 3 is the level Study 1's probe independently selected
for this checkpoint.

"Row-norm PR" is the participation ratio of the squared row norms across the 2,048 rows, as a
fraction of 2,048. "SV k90 / k99" is the number of singular values, out of 64, needed to reach 90% /
99% of the squared Frobenius norm. "SV-PR" is the participation ratio of the squared singular values,
out of 64. Write concentration is accumulated over the whole episode, so it has no "early" row.

| level | pos | row-norm PR | frac rows < 10% max | SV k90 (/64) | SV k99 (/64) | SV-PR (/64) | `x_sparse` zero-frac | write top-1% mass | write top-10% mass |
|---|---|---|---|---|---|---|---|---|---|
| 0 | final | 0.0113 +/- 0.0051 | 0.9604 +/- 0.0234 | 5.42 +/- 2.38 | 15.88 +/- 2.49 | 2.62 +/- 1.02 | 0.8261 +/- 0.0044 | 0.0933 +/- 0.0087 | 0.5607 +/- 0.0365 |
| 0 | early | 0.0665 +/- 0.0166 | 0.7783 +/- 0.0595 | 3.16 +/- 0.83 | 8.03 +/- 1.17 | 2.02 +/- 0.45 | 0.8254 +/- 0.0077 | n/a | n/a |
| 3 | final | 0.0502 +/- 0.0135 | 0.8311 +/- 0.0559 | 10.94 +/- 3.85 | 33.17 +/- 6.49 | 3.05 +/- 1.04 | 0.8108 +/- 0.0246 | 0.0874 +/- 0.0109 | 0.4944 +/- 0.0429 |
| 3 | early | 0.0800 +/- 0.0344 | 0.7507 +/- 0.1067 | 4.24 +/- 1.32 | 13.57 +/- 2.75 | 1.84 +/- 0.35 | 0.8462 +/- 0.0255 | n/a | n/a |
| 5 | final | 0.0211 +/- 0.0081 | 0.9340 +/- 0.0221 | 4.21 +/- 2.38 | 19.68 +/- 4.68 | 1.65 +/- 0.36 | 0.8023 +/- 0.0650 | 0.0939 +/- 0.0210 | 0.3987 +/- 0.0448 |
| 5 | early | 0.0784 +/- 0.0298 | 0.7652 +/- 0.0885 | 3.69 +/- 1.04 | 10.71 +/- 2.39 | 2.11 +/- 0.48 | 0.8464 +/- 0.0746 | n/a | n/a |

**Reading it.**

- **Row sparsity.** At the final timestep, level 3's row-norm participation ratio is 0.050 +/- 0.014
  of 2,048, so about **103 of 2,048 neuron rows** carry the effective mass, and **83% of rows sit
  below 10% of the maximum row norm**. Levels 0 and 5 are sparser still (1.1% and 2.1%). Level 3, the
  level the probe selected, is the least sparse of the three.
- **Effective rank.** At level 3 final, **10.9 +/- 3.9 of 64** singular values carry 90% of the
  squared Frobenius mass and **33.2** carry 99%, with SV-PR 3.05. So sigma is meaningfully but not
  extremely low rank: roughly half the 64-dimensional value space participates non-trivially, while
  the mass within it is concentrated.
- **Activation sparsity is flat and does not explain it.** `x_sparse` is **81% zero** at level 3, and
  80% to 85% zero at every level and position sampled, barely moving while sigma's row and rank
  sparsity swing widely. Activation sparsity alone under-predicts how concentrated sigma's
  accumulated structure becomes.
- **Early to final.** Row occupancy *falls* (level 3: 0.080 to 0.050) while *more* singular values
  pick up non-trivial mass (k90 4.24 to 10.94, k99 13.57 to 33.17). As the episode accumulates, sigma
  concentrates into fewer neuron rows while spreading across more of the value space, not both.
  SV-PR moves far less over the same span, so this is low-energy components picking up mass rather
  than the dominant spectrum flattening.
- **Write concentration.** Accumulated over a whole episode, the **top 1% of neurons take 8.7%** of
  total write mass at level 3 (9.3% to 9.4% at levels 0 and 5) against 1% under a uniform null, and
  the top 10% take 40% to 56% against 10%. Real concentration, roughly 4x to 9x over uniform, but not
  an extreme power law: the bottom 90% of neurons still carry 44% to 60% between them.

**Caveats.** Single checkpoint (gamma = 1.0, seed 0); no cross-seed or cross-gamma comparison, so
these numbers may not generalize. 64 of 2,000 `probe_test` episodes (3.2%), though an 8-episode pilot
matched the 64-episode run to about two decimal places. The "early" position is arbitrary, chosen
only for contrast, not a systematic sweep. "Level" here means the six shared `HBWMCore` transformer
levels.

**Where this bears on an existing argument.** It sharpens, but does not overturn, the estimation
explanation offered post-hoc for the H1 null. At 5% row occupancy roughly 95% of sigma_full's 524,288
columns are near-constant across examples, and the probe's standardization rule (std below 1e-6 maps
to 1.0) neutralizes exactly those, leaving on the order of **26,000 effective features** and about
**88 free parameters per example** rather than the 1,770 obtained by counting all 524,288. The
estimation story is weakened, not eliminated: 88 per example is still a harder regime than any
comparator faced.

That 88 inherits every caveat above. It descends from a row occupancy measured on **one checkpoint,
at one level, over 64 episodes**, so treat the magnitude as indicative and the direction as the point.
The direction survives any plausible revision of the occupancy figure; the exact number does not
deserve the confidence that a two-digit figure invites.

### (b) Probe errors are spatially local

**Method.** Reads only saved probe outputs,
`runs/study1/{bdh_g100_lr0.003,lstm_lr0.003,rwkv_lr0.003}/seed{0,1,2}/probes/*_test.npz`, plus
`data/grid9/probe_test.npz` for the agent-position control. No model inference. n = **41,039 test
rows per seed**, three seeds; the rows and labels are identical across every spec and seed, confirmed
by direct array comparison. BDH levels are the per-seed best from `runs/study1/results/table.json`:
`sigma_full` L3, L3, L4 and `sigma_rownorm` L4, L3, L4.

The chance rate for within-radius-1 is **not** the naive 9/81. It is computed per row from the true
cell's own neighborhood size, which shrinks at edges and corners, then averaged: **0.0969** for
radius 1 and **0.2375** for radius 2. It is identical across specs and seeds because the label
distribution is shared.

| spec | exact acc | within-r1 acc | (chance) | within-r2 acc | (chance) | mean Chebyshev err | mean Manhattan err |
|---|---|---|---|---|---|---|---|
| BDH `sigma_full` | 0.101 +/- 0.007 | 0.308 +/- 0.029 | 0.097 | 0.497 +/- 0.035 | 0.238 | 2.921 +/- 0.133 | 4.005 +/- 0.178 |
| BDH `sigma_rownorm` | 0.172 +/- 0.007 | 0.403 +/- 0.030 | 0.097 | 0.598 +/- 0.030 | 0.238 | 2.442 +/- 0.098 | 3.323 +/- 0.116 |
| LSTM `state_vec` | 0.171 +/- 0.006 | 0.507 +/- 0.015 | 0.097 | 0.703 +/- 0.013 | 0.238 | 2.044 +/- 0.047 | 2.739 +/- 0.062 |
| RWKV `state_vec` | 0.218 +/- 0.007 | 0.542 +/- 0.005 | 0.097 | 0.718 +/- 0.002 | 0.238 | 1.950 +/- 0.008 | 2.619 +/- 0.011 |

Every spec's within-radius-1 accuracy is **3.2x to 5.6x its own chance rate**, and within-radius-2 is
2.1x to 3.0x. Errors are graded, not scattered: even the weakest spec, BDH `sigma_full`, puts
0.101 / 0.207 / 0.189 of its predictions at Chebyshev distance 0 / 1 / 2, so about half its mass lies
within distance 2 of the truth.

**The full predictive distribution, against two nulls.** Expected Chebyshev distance under the
probe's whole output distribution, not just its argmax:

| spec | E[d], actual pairing | E[d], uniform null | E[d], row-shuffled null (20 shuffles) |
|---|---|---|---|
| BDH `sigma_full` | 2.921 | 4.098 | 4.065 |
| BDH `sigma_rownorm` | 2.601 | 4.098 | 4.042 |
| LSTM `state_vec` | 2.576 | 4.098 | 4.032 |
| RWKV `state_vec` | 2.304 | 4.098 | 4.029 |

Every spec beats both nulls, and the row-shuffled null reproduces the uniform null closely, which is
the expected behavior and confirms the shuffle is doing what it should. The *shape* of the output
distribution carries spatial information, not only its peak.

**Agent-proximity control, and it is clean.** If the probe were degenerating to "guess near the
agent", predictions would cluster on the agent's position:

| spec | mean dist(pred, agent) | mean dist(true object, agent) | frac(pred == agent's own cell) |
|---|---|---|---|
| BDH `sigma_full` | 4.005 | 3.810 | 0.008 |
| BDH `sigma_rownorm` | 3.831 | 3.810 | 0.010 |
| LSTM `state_vec` | 3.691 | 3.810 | 0.001 |
| RWKV `state_vec` | 3.712 | 3.810 | 0.001 |

Mean distance from prediction to agent (3.7 to 4.0 cells) sits at the baseline agent-to-object
distance of 3.81, and predictions land on the agent's own cell 0.1% to 1.0% of the time against exact
accuracies of 10% to 22%. The locality tracks the true object, not the agent.

**Caveats.** One environment and one task configuration, not a sweep. "Coarse code" here is read off
distance statistics. The gamma-ablation arms (`bdh_g099`, `bdh_g097`) were not analyzed.

**Where this bears on an existing argument.** The reading recorded in the H1 write-up and in
docs/EXPLAINER.md, that sigma at the probed level may simply carry little precise location
information, is now **half refuted**: an approximate spatial belief is demonstrably present in sigma.
What survives is narrower, that sigma's spatial belief is *blurrier* than the baselines', blurry
enough to fail an exact-cell test. This does not change the H1 verdict, which was about exact-cell
decodability and stands.

### (c) BDH fades toward vagueness where the baselines corrupt

**Method.** Extends (b) using the `bucket` column already written into the same `.npz` files, plus
`hbwm.probes.decisions.h2_curve`, the exact preregistered shape-test function, applied here to a
metric it was never registered for. Bucket row counts are 11,660 / 10,985 / 9,204 / 5,716 / 2,812 /
662, shared across seeds because `probe_test` is one fixed split.

**Pipeline correctness check, passed.** Exact-match accuracy per bucket, recomputed from the raw
`.npz` files, reproduces this document's own H2 table to three decimals
(`np.allclose(atol=1e-3)` is True): BDH `sigma_full` 0.097 / 0.116 / 0.105 / 0.084 / 0.082 / 0.101
and RWKV 0.322 / 0.271 / 0.184 / 0.080 / 0.029 / 0.010, matching the H2 rows above exactly. Bucket
row counts match too.

**Within-radius-1 accuracy by steps since last seen** (mean +/- std over 3 seeds; chance drifts
slightly at 65+ because that bucket is more edge-heavy):

| bucket | n | BDH `sigma_full` | BDH `sigma_rownorm` | LSTM `state_vec` | RWKV `state_vec` | chance |
|---|---|---|---|---|---|---|
| 1-4 | 11660 | 0.292 +/- 0.031 | 0.425 +/- 0.029 | 0.692 +/- 0.007 | 0.686 +/- 0.005 | 0.097 |
| 5-8 | 10985 | 0.343 +/- 0.029 | 0.463 +/- 0.030 | 0.608 +/- 0.021 | 0.650 +/- 0.006 | 0.097 |
| 9-16 | 9204 | 0.321 +/- 0.031 | 0.414 +/- 0.031 | 0.446 +/- 0.020 | 0.517 +/- 0.006 | 0.098 |
| 17-32 | 5716 | 0.283 +/- 0.027 | 0.311 +/- 0.034 | 0.268 +/- 0.019 | 0.329 +/- 0.004 | 0.098 |
| 33-64 | 2812 | 0.264 +/- 0.031 | 0.257 +/- 0.024 | 0.141 +/- 0.009 | 0.155 +/- 0.003 | 0.096 |
| 65+ | 662 | 0.266 +/- 0.031 | 0.283 +/- 0.034 | 0.078 +/- 0.008 | 0.055 +/- 0.013 | 0.085 |

**The ordering inverts.** BDH `sigma_rownorm` starts well behind RWKV (0.425 against 0.686 at bucket
1-4) and ends well ahead of it (0.257 against 0.155 at 33-64, 0.283 against 0.055 at 65+). BDH stays
above its own chance rate at every bucket, including 2.75x chance at 33-64 and 3.13x at 65+.

**Expected distance under the full distribution, against each bucket's own uniform null:**

| bucket | BDH `sigma_full` | BDH `sigma_rownorm` | LSTM `state_vec` | RWKV `state_vec` | uniform null |
|---|---|---|---|---|---|
| 1-4 | 2.993 | 2.466 | 2.123 | 1.908 | 4.115 |
| 5-8 | 2.738 | 2.273 | 2.172 | 1.881 | 4.109 |
| 9-16 | 2.847 | 2.572 | 2.674 | 2.285 | 4.063 |
| 17-32 | 3.016 | 3.017 | 3.229 | 2.913 | 4.027 |
| 33-64 | 3.242 | 3.439 | 3.908 | 3.830 | 4.137 |
| 65+ | 3.555 | 3.675 | **4.609** | **4.850** | 4.542 |

At 65+ both baselines' expected error **exceeds that row set's own uniform null**, and this holds in
**all three seeds individually** (LSTM 4.60, 4.59, 4.63; RWKV 4.87, 4.82, 4.87; the null is a fixed
4.542). A distribution cannot be worse than uniform in expectation unless it places systematic mass
on the *wrong* cells, so the recurrent baselines at very long gaps are **confidently wrong**, not
merely uninformative, plausibly because the recurrent state has been overwritten by more recent
observations. BDH's expected error never crosses its null at any bucket, for either spec.

**Blur-scale calibration.** Predictive distributions were built as a Gaussian blur of the true cell
over Euclidean grid distance, clipped to the 9x9 grid and renormalized, then run through the same
metric pipeline and matched to each spec by E[d]. Overall this puts BDH `sigma_full` at about 3.5
cells, BDH `sigma_rownorm` at about 2.9, LSTM at about 2.8 and RWKV at about 2.4. By bucket, for
BDH's best spec against RWKV:

| bucket | BDH `sigma_rownorm` implied blur | RWKV implied blur |
|---|---|---|
| 1-4 | 2.66 cells | 1.95 cells |
| 5-8 | 2.39 cells | 1.92 cells |
| 9-16 | 2.82 cells | 2.41 cells |
| 17-32 | 3.66 cells | 3.44 cells |
| 33-64 | 5.09 cells | 8.21 cells |
| 65+ | 6.29 cells | not matchable (worse than the uniform null) |

BDH's implied blur widens gradually, from about 2.4 to 2.9 cells at short gaps to 5 or 6 cells at the
longest, still sharper than a uniform guess on a 9-cell-wide grid. RWKV's widens much faster, goes
essentially uniform by 33-64, and becomes literally unmatchable at 65+, because as the blur scale
grows E[d] approaches the uniform null from below and never exceeds it.

**A note on the calibration device itself.** Because a symmetric blur centered on the true cell has
its mode at the true cell, `argmax` recovers the truth for any blur scale, so exact accuracy,
within-r1 accuracy and mean Chebyshev error computed from the argmax are trivially 1.0 / 1.0 / 0.0 at
every scale tested. Only E[d] responds to blur scale. This is a real property of the metrics, not a
bug: **exact-match accuracy is structurally incapable of distinguishing blur scale for a correctly
centered graded code**, which is the whole reason a distribution-aware metric was needed here.

**Study 1's own H2 shape test, applied to the graded metric (exploratory, not preregistered).**
Using `h2_curve` unchanged:

| spec | exact acc graceful | exact ratio 33-64 / 1-4 | within-r1 graceful (exploratory) | within-r1 ratio |
|---|---|---|---|---|
| BDH `sigma_full` | True | 0.846 | True | 0.904 |
| BDH `sigma_rownorm` | True | 0.570 | True | 0.604 |
| LSTM `state_vec` | False | 0.080 | False | 0.204 |
| RWKV `state_vec` | False | 0.090 | False | 0.226 |

The graded metric **flips no verdict**, and for BDH it shows slightly *less* relative decay than
exact-match did. The decay that is real shows up in the distance metrics, not in the accuracy ratio.

**Caveats.** The 65+ bucket has only 662 rows, identical across seeds rather than independently
drawn, so read it as indicative rather than precise; the binomial standard error there is about 0.017
against an observed across-seed std of 0.024 to 0.034, and with three seeds finite-sample noise and
real seed variability cannot be cleanly separated. For **`sigma_full` specifically the decay is
within noise**: the drop from 0.292 at bucket 1-4 to 0.264 at 33-64 is 0.028, about one pooled seed
standard deviation. The decay claim therefore rests on `sigma_rownorm` and on the expected-error and
blur measures, not on `sigma_full` accuracy. The blur calibration uses one isotropic Gaussian noise
model in one environment and is a descriptive device, not a claim that the code is literally
Gaussian-blurred. The "worse than uniform at 65+" result, though consistent across all three seeds,
rests on 662 rows and one environment.

### The headline these three support together

All three architectures hold an **approximate spatial belief**. BDH's is **blurrier at short
horizons**, and that blurriness is what fails H1, which was written as an exact-cell test. But BDH's
belief **degrades toward vagueness while the baselines' degrades into confident error**, and BDH's
expected error never crosses its own uniform null at any horizon while both baselines' does at 65+.

This is post-hoc, single-environment, and not what was preregistered. It does not rescue H1, H3 or
H4, and it does not change the kill criterion's status. It is offered as the most defensible reading
of the saved artifacts and as the finding most worth following up.

## Study 2 preregistration

Decision rules: see README.md, "Preregistration (Study 2)". Preregistration commit:
`be8290e2f38cd30db336761db3aa733b2ad4b2ee`, tagged `study2-prereg`. Study 1's equivalent anchor was
`e674b1da138f905670dde5571e1a1890b134fe36`. Every Study 2 number reported below was produced by runs
started after that commit was pushed to public `main`.

## Study 2 headline

Study 2 is complete. The headline hypothesis **H6 was not supported**, and the preregistered kill
criterion fired. This is a preregistered negative result. The decisions below were taken strictly
under the rules frozen at preregistration commit
`be8290e2f38cd30db336761db3aa733b2ad4b2ee`, before any Study 2 run.

The preregistered consequence applies exactly as written in spec section 7: "if H6 fails against the
LSTM state under matched families, the 'sigma as a linearly or bilinearly readable belief state'
line is closed. Write up and pivot to the imagination study, or abandon." **That line is now
closed.** Study 1 asked whether a flat linear probe can read beliefs out of sigma and got 0.101.
Study 2 gave sigma the readout family that matches how the architecture actually addresses it,
matched on the baselines, and BDH still does not beat the LSTM state by the preregistered margin.
H5 is supported, and H5 is not a rescue: its capacity control, H7, came back
attribute-to-capacity **True**.

### Aggregated results

Pasted from `runs/study1/results2/results.md`, produced by
`uv run python -m hbwm.matrix --phase study2-evaluate --exp study1`. Numbers are verbatim; only the
heading levels are demoted one step so the tables nest under this section.

#### Study 2 probe accuracy (test, best level and hyperparameter per seed; mean over 3 seeds)

| model | family | rank | eff. rank frac | saturated | degenerate | train acc | val acc | test acc | #params | n_train |
|---|---|---|---|---|---|---|---|---|---|---|
| bdh_g100 | derot_flat_linear | - | - | False | False | 0.870 | 0.121 | 0.122 ± 0.011 | 42467328 | 24000 |
| bdh_g100 | derot_query_rank_r | [1, 4, 1] | [0.02, 0.06, 0.02] | False | False | 0.611 | 0.160 | 0.159 ± 0.017 | [684288, 2737152, 684288] | 24000 |
| bdh_g100 | flat_linear | - | - | False | False | 0.695 | 0.104 | 0.101 ± 0.008 | 42467328 | 24000 |
| bdh_g100 | mlp_randproj | - | - | False | False | 0.823 | 0.127 | 0.121 ± 0.013 | 2138624 | 24000 |
| bdh_g100 | mlp_rownorm | - | - | False | False | 0.741 | 0.141 | 0.144 ± 0.012 | 4235776 | 24000 |
| bdh_g100 | mlp_state | - | - | False | False | 0.726 | 0.104 | 0.101 ± 0.007 | 268476928 | 24000 |
| bdh_g100 | query_rank_r | [16, 1, 4] | [0.25, 0.02, 0.06] | False | False | 0.548 | 0.141 | 0.135 ± 0.019 | [10948608, 684288, 2737152] | 24000 |
| bdh_g100 | shared_query_rank_r | [4, 1, 16] | [0.06, 0.02, 0.25] | False | False | 0.539 | 0.145 | 0.141 ± 0.011 | [115712, 28928, 462848] | 24000 |
| lstm | flat_linear | - | - | False | False | 0.298 | 0.115 | 0.116 ± 0.005 | 113400 | 24000 |
| lstm | mlp_state | - | - | False | False | 0.492 | 0.144 | 0.146 ± 0.006 | 758272 | 24000 |
| lstm | query_rank_r | [16, 16, 4] | 1.00 | True | False | 0.283 | 0.113 | 0.113 ± 0.006 | [458784, 458784, 114696] | 24000 |
| lstm | shared_query_rank_r | [4, 4, 16] | 1.00 | True | False | 0.296 | 0.112 | 0.113 ± 0.004 | [113416, 113416, 453664] | 24000 |
| rwkv | flat_linear | - | - | False | False | 0.415 | 0.148 | 0.148 ± 0.005 | 285120 | 24000 |
| rwkv | mlp_state | - | - | False | False | 0.602 | 0.157 | 0.165 ± 0.006 | 1843712 | 24000 |
| rwkv | query_rank_r | 16 | 0.80 | False | False | 0.344 | 0.143 | 0.145 ± 0.003 | 254016 | 24000 |
| rwkv | shared_query_rank_r | 16 | 0.80 | False | False | 0.370 | 0.143 | 0.142 ± 0.003 | 228416 | 24000 |

Degeneracy criterion (preregistered, spec 7): an arm is degenerate, and excluded from H6's best-matched-family selection, if its training accuracy exceeds 0.95 while its validation accuracy stays below 0.022 (twice the majority-class chance rate 0.011) at every L2 value. Degenerate arms are still fitted and still reported above with their parameter count and training accuracy, so the call can be audited. `mlp_state` on BDH has 268,476,928 weights in total against 24,000 training examples, which is the number to read that row by. Of those, the first layer alone is 524,288 x 512 = 268,435,456 weights (spec 5.2); the remainder is the 512 x n_classes output layer of Appendix B's count.

Family 5 membership: BDH carries `mlp_state` plus the two BDH-only reductions `mlp_rownorm` and `mlp_randproj`; each baseline carries `mlp_state` alone, because row norms of a 4 x 350 or 20 x 176 reshape are not a control and projecting 1,400 up to 4,096 is an expansion. H6's family 5 comparison is therefore `mlp_state` against `mlp_state` against `mlp_state`; the reductions are context and feed H7.

Random projection (spec 4.5, `mlp_randproj`): a FIXED, not-learned sparse sign matrix with 64 nonzeros per output dimension, drawn with signs [-1, 1] from seed 0, mapping the standardized_flat_sigma to 4096 dimensions. Recorded per run in `probes2/done.json`; read here from bdh_g100/seed0.

#### H5 (format and estimation) supported: **True**

Best structured family `derot_query_rank_r` at 0.159 against `flat_linear` at 0.101; mean diff +0.058, paired diffs [0.066, 0.043, 0.067].

#### H6 (headline) supported: **False**, carried by `derot_query_rank_r`

| comparator | mean | mean diff | paired diffs | passes | saturated arm |
|---|---|---|---|---|---|
| lstm | 0.113 | +0.046 | [0.042, 0.031, 0.065] | False | True |
| rwkv | 0.145 | +0.014 | [0.02, -0.008, 0.029] | False | False |

Kill criterion fired: **True**. Rank-constraint artifact warning: **False**.

Families eligible for H6 after the degeneracy criterion: `derot_flat_linear`, `derot_query_rank_r`, `flat_linear`, `mlp_state`, `query_rank_r`, `shared_query_rank_r`. Excluded as degenerate: none.

#### H7 (attribution, gates nothing)

Verdict: `mlp_rownorm` at 0.144 against `derot_query_rank_r` at 0.159; attribute to capacity: **True**.

Context (BDH-only, decides nothing): `mlp_randproj` at 0.121 against `derot_query_rank_r` at 0.159; attribute to capacity: **False**.

#### H8 (belief revision, clock rebaselined to the first not-visible step)

| model | mean frac(latency <= 5) | excluded n | episodes n | excluded n per seed | mean excluded frac | low coverage | supported |
|---|---|---|---|---|---|---|---|
| bdh_g100 | 0.771 | 51 | 1866 | [17, 17, 17] | 0.027 | False | True |
| lstm | 0.969 | 51 | 1866 | [17, 17, 17] | 0.027 | False | True |
| rwkv | 0.976 | 51 | 1866 | [17, 17, 17] | 0.027 | False | True |

#### Cross-study bridge rows (continuity only, decide nothing)

| model | flat_linear acc | n_train pairs |
|---|---|---|
| lstm | 0.170 | 61400 |
| rwkv | 0.218 | 61400 |

### Decisions under the preregistered rules

- **H6 (the headline): not supported. Kill criterion fired.** The rule is a 5-point mean margin over
  *both* baseline states with all three paired-by-seed differences positive, within the best matched
  family. The best matched family is `derot_query_rank_r` at 0.159 on BDH. Against the LSTM state
  (0.113) the mean difference is +0.046, which is below the 5-point bar even though all three paired
  differences are positive ([0.042, 0.031, 0.065]). Against the RWKV state (0.145) the mean
  difference is +0.014, below the bar, and one paired difference is **negative** (-0.008 on seed 1),
  so that comparison fails on both counts. The preregistered kill condition is "H6 fails against the
  LSTM state under matched families", which is exactly what happened. The line is closed and the
  preregistered response is to write up and pivot, not to keep tuning readouts.
- **H6 read against the reshape-free families, as spec 5.2 and the H6 rule require.** The LSTM's
  factorized arms are **rank-saturated**: its 4 x 350 reshape saturates at r = 4, and the
  preregistered grid's r = 4 and r = 16 both sit at or past that, so `query_rank_r` and
  `shared_query_rank_r` on the LSTM carry effective rank fraction 1.00 and are `flat_linear` in a
  different parameterization. The rule therefore requires the reading to be checked against
  families 1 and 5, whose baseline arms cannot saturate. That reading is **worse for BDH, not
  better**: on `flat_linear`, BDH 0.101 against lstm 0.116 and rwkv 0.148; on `mlp_state`, BDH 0.101
  against lstm 0.146 and rwkv 0.165. BDH loses both reshape-free comparisons outright. The
  `artifact_warning` flag is **False**, and it is False only because BDH did not win: the saturation
  caveat exists to qualify a BDH win over a saturated baseline arm, and there was no win to qualify.
  It is not an all-clear.
- **Eligibility and degeneracy: nothing was excluded.** All six matched families were eligible for
  H6's selection: `derot_flat_linear`, `derot_query_rank_r`, `flat_linear`, `mlp_state`,
  `query_rank_r`, `shared_query_rank_r`. The preregistered degeneracy criterion (train accuracy
  above 0.95 with validation accuracy below 0.022 at every L2 value) excluded **no arm**. In
  particular `mlp_state` on BDH, the arm the criterion was written for, did not trip it: its 268.5 M
  parameters against 24,000 examples reached train 0.726 and val 0.104, so it decided H6 on the same
  footing as every other arm. Every arm's parameter count, `n_train`, and training accuracy are in
  the table above so the call can be audited rather than taken on trust.
- **H5 (format and estimation): supported, and it must not be read alone.** The best structured
  sigma readout, `derot_query_rank_r` at 0.159, beats `flat_linear` on sigma at 0.101 by +0.058,
  clearing the 5-point margin, with all three paired-by-seed differences positive
  ([0.066, 0.043, 0.067]). Supported means Study 1's H1 failure was at least partly an artifact of
  readout format or parameter estimation. It does **not** mean the gain is about associative,
  query-addressed structure. The spec's own risk table requires this to be stated here: **H5 alone
  cannot separate format from estimation efficiency**, and H7 is the control that separates them.
  The parameter counts are the most likely reason: the winning arm fits 0.68 M to 2.74 M parameters where
  `flat_linear` fits 42.5 M, on the same 24,000 examples.
- **H7 (attribution, gates nothing): attribute to capacity, True.** `mlp_rownorm` reaches 0.144
  against the best structured readout's 0.159, a gap of -0.015, inside the preregistered 2-point
  tolerance, so the rule attributes the gain to capacity and nonlinearity rather than to associative
  structure. This is the decisive qualifier on H5. `mlp_rownorm` is a plain MLP on
  rotation-invariant row norms with **no query structure at all**, and it lands within 1.5 points of
  the best bilinear readout on 4.24 M parameters. Read together, as the rules require, H5 and H7 say
  that the +0.058 over the flat probe is attributable to capacity and nonlinearity and is **not**
  demonstrated to be about query-addressed associative structure. For context, and deciding nothing:
  `mlp_randproj`, the fixed sparse-sign projection, reaches 0.121 against 0.159, a gap of -0.038,
  and does not attribute to capacity, so the row-norm reduction outperforms a fixed sparse-sign
  projection at half the reduced width (8,192 to 512 against 4,096 to 512, 4.24 M parameters
  against 2.14 M). The two are not width-matched, so this comparison decides nothing.
- **H8 (belief revision): supported for all three models, and it reverses Study 1's H3.** Under the
  preregistered clock, latency is measured from t0, the first step at or after re-observation at
  which the object is **not** visible. The fraction of episodes revising within 5 steps is
  bdh_g100 **0.771**, lstm 0.969, rwkv 0.976, all above the 0.70 bar. Study 1's H3 measured
  bdh_g100 at 0.157 and concluded BDH does not revise its beliefs; that conclusion was an artifact
  of the clock. Study 1 started counting at the re-observation step itself, while the answer was
  still inside the agent's 3x3 window, so it charged BDH for steps on which no revision was
  measurable. **This changes the Study 1 conclusion about belief revision.** It does not make BDH
  competitive: 0.771 still trails both baselines substantially (0.969 and 0.976), so the honest
  reading is that BDH revises, but more slowly and less reliably than either baseline. Coverage:
  51 of 1,866 moved-and-re-observed episodes (2.7%, 17 per seed) have no t0 because the agent stays
  adjacent to the moved object through the end of the episode; those are excluded from the
  denominator, not counted as failures, and the 25% low-coverage flag did **not** trip.
- **Reproduction checks: Study 2 measures what Study 1 measured.** Two independent checks, neither
  of which decides anything. First, BDH's `flat_linear` arm, which is Study 1's `sigma_full` probe
  refit under Study 2's own machinery (restarts, a `--families` code path, a fresh trainer), lands
  at **0.101** (0.100815), reproducing Study 1's 0.101 to the reported precision. Second, the cross-study bridge rows refit
  `flat_linear` on both baseline states at the full 61,400-pair budget and give lstm **0.170** and
  rwkv **0.218**, against Study 1's 0.171 and 0.218. The two studies' numbers are therefore directly
  comparable and the H6 negative is not an artifact of new machinery. **The bridge rows are
  continuity only and are used by no decision rule**; H6 remains the matched 24,000-pair comparison.
- **Selection budget, so the reader can discount it.** Rank, L2, restart, level and family were all
  selected on `probe_val`, which is 1,000 held-out episodes disjoint from both `probe_train` and
  `probe_test`. Every number reported above is on `probe_test`. Selecting five things on validation
  inflates the winner; the reproduction checks above are the check that the inflation is not
  carrying the result.

### Exploratory: descriptive structure of sigma (spec 4.8)

**These decide nothing.** No number below is a criterion, none of them can support or refute H5 to
H8, and none of them was preregistered. They exist because Study 1 never actually answered whether
sigma is sparse: H4 asked the narrower question of whether the *decodable* signal concentrates in
few features, and its negative (median k90 = 524,288, the full feature count) is entangled with H1's
negative, because the feature ranking it used came from a probe that only reached 0.101. "No sparse
decodable subset" and "nothing decodable to concentrate" are not separated by that result.

Measured on the four BDH checkpoint-levels Study 2 probes, on a fixed seeded 1,024-row subsample of
the cached `probe_train` features per checkpoint-level, on the **unstandardized** state. Every cell
is a median over that subsample; per-cell 10th and 90th percentiles are in
`runs/study1/results2/structure.json` and in the per-run
`runs/study1/*/seed*/probes2/sigma_structure_L*.json`.

| checkpoint-level | row-norm PR (of 2048) | frac rows < 1% of max | 90% comps (/64) | 99% comps (/64) | SV-PR (/64) | write top 1% | write top 10% | ReLU zero frac | atlas max-share | atlas norm. entropy | frac max-share > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0 L3 | 110.9 | 0.182 | 10 | 32 | 2.78 | 0.089 | 0.478 | 0.832 | 0.181 | 0.748 | 0.191 |
| seed0 L4 | 103.1 | 0.151 | 8 | 29 | 2.37 | 0.084 | 0.426 | 0.844 | 0.159 | 0.793 | 0.114 |
| seed1 L3 | 73.3 | 0.238 | 9 | 32 | 2.17 | 0.111 | 0.555 | 0.829 | 0.234 | 0.692 | 0.224 |
| seed2 L4 | 104.8 | 0.113 | 10 | 30 | 3.00 | 0.067 | 0.413 | 0.797 | 0.165 | 0.808 | 0.055 |

Read across the four checkpoint-levels:

- **Row-norm sparsity.** The participation ratio of the squared row norms is 73 to 111 effective
  neurons out of 2,048, so roughly 3.5 to 5.5 percent of the rows carry the mass. The fraction of rows
  below 1% of the row-norm max is 0.11 to 0.24.
- **Effective rank.** 8 to 10 singular components reach 90% of the squared Frobenius mass and 29 to
  32 reach 99%, out of a maximum of 64; the spectral participation ratio is 2.2 to 3.0.
- **Write concentration.** The top 1% of rows (20 of 2,048) receive 0.067 to 0.111 of the total
  write mass and the top 10% receive 0.41 to 0.56.
- **Activation sparsity.** The ReLU zero fraction of `x_sparse` at the same timesteps is 0.80 to
  0.84. This is a property of the write key, not of the accumulated state, and is reported as
  contrast only.
- **Atlas selectivity.** Median max-share over the token-conditional activation profile is 0.159 to
  0.234 and median normalized entropy is 0.69 to 0.81, with only 0.055 to 0.224 of live neurons
  exceeding a max-share of 0.5. A concept-aligned basis would show a heavy tail of low-entropy,
  high-max-share neurons. A minority of roughly 6 to 22 percent of live neurons are single-token
  dominated, and the p90 of max-share reaches 0.757 at seed0 L3, so the tail is not empty; but the
  bulk of the distribution is not concept-aligned.
  The normalized-entropy denominator is ln(29), not ln(33): 29 of the 34 vocabulary entries
  actually occur in the 500-episode atlas sample. The five that never occur are `PAD` plus the four
  coordinate slots the 34-token vocabulary reserves for grids up to 11 wide (x = 9, 10 and y = 9,
  10), which a 9x9 grid never reaches. Spec 4.8 anticipated 33 by accounting for `PAD` alone, so
  this is a correction to the spec's arithmetic rather than a deviation in the measurement, and it
  changes nothing preregistered.

**The honest reading.** Sigma **is** structurally sparse and low rank: a few percent of rows carry
the mass, and 8 to 10 components out of 64 carry 90% of it. But its neuron basis is largely
**distributed rather than concept-aligned**: most neurons spread their activation broadly across the
vocabulary, and only a small minority are dominated by a single token. That is the question H4 could
not separate, and it is answered here in a way that is consistent with everything above. Structural
sparsity is real; concept-alignment is not, and that is consistent with why sparsity alone never
made sigma readable. This is exploratory and cannot revise H4, which stands as reported.

The preliminary version of measurements 1 to 4, run before this implementation existed on the single
checkpoint `runs/study1/bdh_g100_lr0.003/seed0/ckpt.pt` at levels 0, 3 and 5, is reported above
under "Post-hoc analyses (exploratory, not preregistered) (a) Sigma is structurally sparse and
lowish rank" and is labeled there with its checkpoint, levels and method. It is a different sample
and a different normalization from the table here and the two are not interchangeable; the table
here is the specified measurement.

## Post-hoc follow-ups after Study 2 (exploratory, not preregistered)

**These decide nothing, and nothing here is a rescue.** None of the four findings below was
preregistered. None of them can support, refute or qualify H1 to H8, and none of them reopens a
threshold. Study 2's H6 kill criterion fired, and the preregistered consequence stands exactly as
written in spec section 7: the "sigma as a linearly or bilinearly readable belief state" line is
**closed**. Nothing measured here reopens it. What these four do is characterize what is in sigma
now that the readability question has been answered in the negative, which is a different question
from the one the preregistration asked.

Every number below is reproducible from a named script and, where one exists, a named JSON.

### The objective did demand memory, and BDH is twice as good as the LSTM at supplying it

A tempting reading of Study 1's and Study 2's negatives is that the task never rewarded remembering
where an out-of-view object was, so there was nothing for a probe to find. That reading is wrong,
and this is the measurement that rules it out.

The first half is data only, no model and no GPU: `analysis/posthoc/memory_demand_of_objective.py`
over all 2,000 `probe_test` episodes. It writes stdout, not a JSON.

| quantity | value |
|---|---|
| window cells predicted per episode | 873 (9 per observation, 97 observations) |
| window cells that are empty | 0.8430 |
| window cells that are wall | 0.1245 |
| window cells that are objects | 0.0325 |
| of object-visible events: still in view from last step | 0.6949 |
| of object-visible events: returning after an absence | 0.2133 |
| of object-visible events: first sighting | 0.0919 |
| returns where the object is STILL in the cell last seen | 0.9682 |
| window cells where memory of an absent object could help | 11,716 of 1,746,000 = 0.671% |
| the same, as a share of all 2,326,000 next-token predictions | 0.504% |
| the same, restricted to absences of 9 or more steps | 5,404 = 0.310% of window cells |
| absence length at the moment of return | median 8, p75 16, p90 32, max 94 steps |

So memory of an absolute cell can pay off on 0.67% of window cells and 0.50% of predicted tokens,
and 96.8% of returns are to the exact cell the object was last seen in, which is what makes a
remembered cell worth anything at all.

A token count is not a loss share. `analysis/posthoc/memory_loss_attribution.py` does one forward
pass per checkpoint over `probe_test`, 3 seeds per model, no training and no probes, and splits the
residual cross entropy by token class. Mean over 3 seeds, in nats:

| model | overall CE | memory-relevant cells | share of total loss | object cells needing no memory |
|---|---|---|---|---|
| bdh_g100 | 0.02458 | 0.6832 | 0.1515 | 0.5778 |
| lstm | 0.02910 | 1.3555 | 0.2556 | 0.5949 |
| rwkv | 0.02423 | 0.6643 | 0.1504 | 0.5811 |

Those cells are about half a percent of the tokens and carry 15 to 26 percent of the total loss, so
the objective did press on them. And **BDH is roughly twice as good as the LSTM on exactly the
predictions that require the memory** (0.683 against 1.356) while the three models are
indistinguishable on object cells that need no memory (0.578, 0.595, 0.581). Whatever separates BDH
from the LSTM here is specific to the memory-requiring predictions.

Yet Study 2 found the LSTM's state **more** linearly decodable for position than BDH's sigma under
the matched `mlp_state` family: lstm 0.146 against bdh_g100 0.101. **Decodability and functional
use are dissociated in the direction that matters.** The model with the less decodable state uses
its state better on the very tokens the decoding target was about. This is a concrete instance of
the standard critique of probing: a probe's failure is evidence about the probe-model pair, not
about the model. It does not soften H6, whose criterion was decodability and which failed on
decodability.

### Causal per-level activation patching of the associative read

Probes are correlational. This is the causal test, from `analysis/posthoc/ykv_causal_patch.py`,
reported in `runs/study1/results2/posthoc_ykv_causal.json`. On the loaded model instance only, and
without editing `hbwm/`, the bound method `HBWMCore._attend` is monkeypatched so its return value
during the parallel forward is replaced by the SAME tensor rolled one step along the batch axis.
Each episode then receives a different episode's associative read: same marginal distribution, same
norms, wrong content. 2,000 `probe_test` episodes, 3 seeds, `bdh_g100_lr0.003`, chunked at 25
episodes so the roll is within a chunk.

**Design note on why the intervention had to change content.** `self.ln` in
`hbwm/bdh/upstream/bdh.py` is `nn.LayerNorm(D, elementwise_affine=False, bias=False)` and is
therefore scale invariant. Scaling `_attend`'s output by a constant is a no-op and would have been a
useless intervention.

Token classes: c1 = window cell that is empty or wall, c3 = window cell holding an object that needs
no memory (the control), c2 = window cell holding an object returning after an absence to the same
cell it was last seen in (11,716 tokens, 0.55% of the 2,134,000 loss-masked tokens scored here).
Deltas are against the intact model, whose overall CE is 0.0246 and whose class CEs are c1 0.0057,
c2 0.6832, c3 0.5778. Mean over 3 seeds; `ratio` is the ratio of the mean deltas (`ratio_of_means`
in the JSON), and `excess` is d c2 minus d c3.

| condition | overall CE | d c1 | d c3 (control) | d c2 (memory) | excess | ratio | saturated |
|---|---|---|---|---|---|---|---|
| patched_L0 | 2.7652 | +1.2960 | +7.0340 | +9.1765 | +2.1425 | 1.30 | **yes** |
| patched_L1 | 1.2108 | +0.7892 | +4.9569 | +7.3631 | +2.4062 | 1.49 | **yes** |
| patched_L2 | 0.4984 | +0.1570 | +5.1246 | +5.9173 | +0.7927 | 1.15 | no |
| patched_L3 | 0.1651 | +0.0236 | +4.3081 | +4.8766 | +0.5685 | 1.13 | no |
| patched_L4 | 0.0892 | +0.0020 | +1.7645 | +3.9824 | +2.2179 | 2.26 | no |
| patched_L5 | 0.0522 | -0.0014 | +0.4852 | +2.5948 | +2.1096 | 5.35 | no |
| all levels patched | 4.2927 | +2.8992 | +10.6903 | +10.2810 | -0.4092 | 0.96 | **yes** |
| all levels zeroed | 3.3563 | +3.0968 | +2.6765 | +2.6050 | -0.0715 | 0.97 | **yes** |

**The saturated arms are discarded, not reported as nulls.** Above an overall CE of about 1 nat both
object classes sit near a floor that is worse than uniform (uniform over the 34-token vocabulary is
ln(34) = 3.53 nats, i.e. confidently wrong), so the c2 against c3 contrast cannot discriminate and
its ratio is a ceiling artifact. That applies to `patched_L0`, `patched_L1` and both all-levels
arms. The two all-levels arms are retained only as a saturation reference, and what they show is
that the associative read is causally load-bearing for essentially every prediction, including
trivially predictable empty and wall cells (d c1 of +2.90 and +3.10). That is not a finding about
memory.

**The result is the deep-level selectivity.** Patching L5 alone leaves the model nearly intact:
overall CE moves from 0.0246 to 0.0522, empty and wall cells are unchanged at -0.0014, and control
object cells take +0.4852. On memory-relevant cells the same patch costs **+2.5948 nats**, a 5.35x
selectivity over the control. Those cells are 0.55% of scored tokens and account for **51.6% of all
the extra nats that patch produces** (30,401 of 58,945 excess nats); at L4 the same share is 33.9%.
This is causal evidence rather than correlational: the memory is carried by sigma's associative
read, and it is concentrated in the deep levels.

Note that the per-seed ratios are much more dispersed than the ratio of means: averaging the three
per-seed ratios gives 4.34 at L4 and 16.77 at L5, because seed 0 and seed 2 have near-zero control
damage at L5. The table reports the ratio of the mean deltas, which is the conservative summary.

**Convergence with the Study 2 probe, and where it breaks.** Taking the most selective level per
seed as the unsaturated level with the largest `excess`, the answer is L4, L5, L4 for seeds 0, 1, 2.
Study 2's sigma probe selected best-on-validation levels [3, 4], [3] and [4] for the same seeds.
Pooled over seeds the most selective level is L4, which is inside the probe's [3, 4]: they agree.
Per seed, seeds 0 and 2 agree and **seed 1 does not** (patching says L5, the probe chose L3). The
JSON records this as `agreement_per_seed` {0: true, 1: false, 2: true} with `agreement_pooled` true.
The divergence is reported, not rounded away. Note also that under the alternative definition of
"most selective" as the largest ratio rather than the largest excess, L5 wins in all three seeds, so
the pooled convergence with the probe depends on the summary chosen.

### Sigma is massively superposed

Spec 4.8 measurement 4 defines a cancellation index that the shipped Study 2 code never reported.
`analysis/posthoc/cancellation_index.py` computes it, writing
`runs/study1/results2/posthoc_cancellation.json`. The index is sum(a) / sum(w), where a[h,n] is the
squared norm of sigma row (h, n) at the sampled timestep and w[h,n] is the accumulated squared write
mass routed into that row up to that timestep. 1,024 sampled `probe_train` examples per
checkpoint-level, batched at 32 episodes, on the same four BDH checkpoint-levels Study 2 probes.

**Validation.** The script recomputes two quantities already published in
`runs/study1/results2/structure.json`, the row-norm and write-mass participation ratios, and
reproduces all four checkpoint-levels to under 1e-6 relative error (worst case 7.6e-8). That is what
establishes that its sampling, its rope-reconstructed query, and its accumulator ordering match the
pass `measure_sigma_structure` actually ran.

| checkpoint-level | index median [p10, p90] | top 1% rows | top 10% rows | Pearson r(w,a) | Spearman rho(w,a) | writes per row (median) |
|---|---|---|---|---|---|---|
| seed0 L3 | 6.17 [5.56, 7.16] | 1.21 | 1.70 | 0.16 | 0.75 | 147 |
| seed0 L4 | 9.47 [7.37, 11.01] | 3.80 | 6.11 | 0.22 | 0.65 | 165.5 |
| seed1 L3 | 4.90 [4.38, 5.74] | 0.52 | 1.76 | 0.11 | 0.79 | 142 |
| seed2 L4 | 10.73 [7.81, 13.91] | 8.90 | 6.46 | 0.25 | 0.59 | 214 |

**The arithmetic.** An index of 1 is what statistically independent writes would give, since their
squared norms add. Perfect alignment of every write into a row would give an index equal to the
number of writes into that row. With 142 to 214 writes per row and an index of only 4.9 to 10.7,
only about 5 to 10 writes' worth of mass survives coherently: the ratio of realized to
perfectly-aligned mass is 0.034 to 0.050, so **roughly 95% of everything routed into sigma is
destroyed by interference**.

The heavily written rows are the worst, not the best. The top-1% index is below the pooled median at
**all four** checkpoint-levels, and at seed1 L3 it is 0.52, below 1, meaning those rows are actively
cancelling rather than merely failing to add. The Pearson r of 0.11 to 0.25 against a Spearman rho
of 0.59 to 0.79 says the same thing from another angle: realized row mass tracks routed write mass
in rank order but not in magnitude, which is the signature of interference, not routing, setting the
scale.

`bdh_g100` runs at gamma = 1.0, confirmed as `decay_gamma` in every record, so nothing is ever
forgotten and every write from the whole 1,164-step episode is still present to interfere.

### Probing yKV, the model's own associative read: a negative result that refutes the leading hypothesis

Nothing in Study 1 or Study 2 had ever probed `yKV`, the OUTPUT of the model's own associative read.
Study 1 probed `sigma_full`, `sigma_rownorm`, `x_sparse` and `resid`; Study 2 probed reductions and
readouts of sigma. All of them learn a STATIC query. The architecture reads sigma with an
input-dependent, rope-rotated query, so the leading hypothesis after H6 failed was that position
would be most linearly present in what that query actually extracts.

`analysis/posthoc/ykv_probe.py`, reported in `runs/study1/results2/posthoc_ykv_probe.json`. The
feature is `yKV = sum_n rope(relu(x_t @ W_enc), t)[h,n] * sigma[h,n,d]`, flattened over
(n_head, n_embd), 256 dimensions, standardized, with a flat multinomial linear probe. Protocol
matched to Study 2: 24,000 train, 20,244 val and 41,039 test pairs, 81 classes, L2 grid
[1e-4, 1e-3, 1e-2, 1e-1], 20 epochs, lr 1e-3, batch 512, selected on `probe_val`, reported on
`probe_test`, all 6 levels, 3 seeds. Chance is 0.011.

| level | mean test accuracy over 3 seeds |
|---|---|
| L0 | 0.0325 |
| L1 | 0.0402 |
| L2 | 0.0481 |
| L3 | 0.0540 |
| L4 | 0.0574 |
| L5 | 0.0558 |

| seed | best level | best L2 | test accuracy | bootstrap 95% CI |
|---|---|---|---|---|
| 0 | L4 | 1e-4 | 0.0538 | [0.0508, 0.0570] |
| 1 | L4 | 1e-3 | 0.0469 | [0.0442, 0.0498] |
| 2 | L4 | 1e-4 | 0.0715 | [0.0679, 0.0750] |

Mean at the best level 0.0574 (sd 0.0104 over the 3 seeds, as recorded in the JSON; the sample sd
with Bessel's correction is 0.0127). Reference points, all from Study 2's table above:
`flat_linear` on sigma 0.101, the best structured readout `derot_query_rank_r` 0.159, and the LSTM
state's `mlp_state` 0.146.

**The hypothesis is not supported.** yKV decodes position at about 5x chance, but roughly **half as
well as raw sigma** and about a third as well as the best structured probe. Reading sigma the way
the circuit reads it DISCARDS linearly decodable position rather than concentrating it.

This is not an estimation artifact, and that is the point of choosing this feature. yKV is 256
dimensions against 24,000 training pairs, so unlike the 524,288-dimensional flat probe on sigma this
one is comfortably well conditioned. H5's estimation-efficiency explanation therefore does not
apply, and the low number means genuine absence rather than a fitting failure.

One corroboration is worth recording: yKV peaks at L4 for every seed, which is the same level the
causal patching found most selective pooled over seeds.

### What the four say together

Composed honestly, and stated as an inference rather than a demonstration:

- The information about where an out-of-view object is **is causally load-bearing**. Corrupting the
  deep-level associative read costs +2.59 nats on exactly the tokens that need it while leaving the
  rest of the model nearly intact, and those tokens absorb half the damage (finding 2).
- It is **functionally used, and used better than the LSTM uses its own**. BDH halves the LSTM's
  loss on the memory-requiring predictions while matching it everywhere else (finding 1).
- It is **not linearly decodable anywhere tested**: not in sigma (0.101), not in sigma's row norms
  under a linear readout, and not in the circuit's own extraction of sigma (0.057, finding 4), even
  though the last of those is a well-conditioned 256-dimensional problem.
- Finding 3 supplies a **candidate mechanism**. With gamma = 1.0 and 142 to 214 writes per row,
  sigma is superposed to the point where roughly 95% of routed mass is destroyed by interference,
  and the most heavily written rows cancel the hardest. What survives is a thin coherent residue
  that the model's own input-dependent query can exploit at the moment it reads, but that no single
  fixed direction can expose.

That last point is a **hypothesis consistent with the measurements, not something the measurements
establish**. Nothing here tests it directly. What the four together do support is that the
representation looks **genuinely nonlinear and distributed**, which is a stronger and different
claim from the format hypothesis Study 2 tested and rejected: H5 and H7 said the readout format
matters somewhat but attributes to capacity, and these findings say the underlying encoding may not
have a readable format at all. It remains an inference. Testing it would need an intervention that
manipulates interference directly, for example a gamma sweep or a capacity sweep with the
cancellation index as the dependent variable, and none of that was run.

**None of this revises H1 to H8.** H6 is not supported, its kill criterion fired, and the
linearly-or-bilinearly-readable-belief-state line stays closed.
