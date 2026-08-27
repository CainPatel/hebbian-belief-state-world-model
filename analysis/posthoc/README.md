# Post-hoc analysis scripts

Three exploratory, descriptive scripts run **after** Study 1 closed. Their findings are written up in
[RESULTS.md](../../RESULTS.md) under "Post-hoc analyses (exploratory, not preregistered)" and
summarized in [docs/EXPLAINER.md](../../docs/EXPLAINER.md) section 8.

**They decide nothing.** None of them was preregistered, none changes a preregistered decision, and
none reopens a threshold. H1 remains not supported with the kill criterion fired, H2 remains a pass
for `gamma = 1.0` only, and H3 and H4 remain not supported for BDH. These scripts exist so the
post-hoc numbers are reproducible, not so they can be promoted to findings.

**They are read-only on `runs/` and `data/`.** Nothing here writes into either tree, retrains a
model, or mutates a checkpoint. Two of the three do no model inference at all; the third only drives
saved weights forward through the recorder. Each writes one JSON or one block of stdout, and nothing
else.

## What each one does

| script | what it measures | model inference? | output |
|---|---|---|---|
| `sigma_structure.py` | How sparse and how low-rank `sigma` is. Row-norm participation ratio, singular-value k90 / k99 / participation ratio, `x_sparse` zero fraction, and per-neuron write-mass concentration, at levels 0, 3 and 5 and at an early and a final timestep. | Yes. Drives one checkpoint through `SigmaRecorder`. | stdout |
| `spatial_locality.py` | Whether the probe's errors are spatially *local* rather than scattered. Within-radius-1 and -2 accuracy against a per-row chance rate, Chebyshev and Manhattan error, expected distance under the full predictive distribution against a uniform and a row-shuffled null, marginal x and y decoding, and an agent-proximity control. | No. Reads saved probe `.npz` only. | `spatial_locality_results.json` |
| `spatial_locality_buckets.py` | Whether that locality decays with steps since last seen. Everything in `spatial_locality.py` broken down by the six `steps_since_seen` buckets, plus Study 1's own `h2_curve` shape test applied to the graded metric, plus a Gaussian blur-scale calibration. Also recomputes exact-match accuracy per bucket as a correctness check against RESULTS.md's H2 table. | No. Reads saved probe `.npz` only. | `spatial_locality_buckets_results.json` |

`spatial_locality_buckets.py` builds on `spatial_locality.py` and uses the identical spec selection,
so run them in that order if you want both outputs to agree.

## Running them

All three expect the Study 1 worktree layout: a directory containing `runs/study1/...`,
`data/grid9/...` and an importable `hbwm/`. In this repository those artifacts are gitignored and
live in the sibling worktree `.claude/worktrees/study1-impl/`, which is the default.

```sh
# sigma_structure.py resolves runs/ and data/ relative to the WORKING DIRECTORY
# (the checkpoint's own cfg.data_dir is "data/grid9"), so cd into the worktree
# first and point at the script by path.
cd .claude/worktrees/study1-impl
uv run python ../../../analysis/posthoc/sigma_structure.py

# The two spatial scripts take their root from HBWM_ROOT, which defaults to that
# same worktree, so they run from anywhere. HBWM_POSTHOC_OUT chooses where the
# JSON lands and defaults to the working directory.
HBWM_POSTHOC_OUT=/tmp uv run python analysis/posthoc/spatial_locality.py
HBWM_POSTHOC_OUT=/tmp uv run python analysis/posthoc/spatial_locality_buckets.py
```

`sigma_structure.py` needs room for one checkpoint plus a 64-episode batch of `sigma`; it prefers MPS
and falls back to CPU, and takes about 25 s wall time (about 19 s of that is the recorder pass). The
two spatial scripts are pure NumPy over saved arrays and take seconds.

Verified after the move: all three reproduce their reported numbers from these paths, and the two
JSON outputs are byte-identical to the originals the RESULTS.md tables were written from.

## Hard-coded choices worth knowing about

- `sigma_structure.py` reads exactly one checkpoint, `runs/study1/bdh_g100_lr0.003/seed0/ckpt.pt`,
  and the first 64 of the 2,000 `probe_test` episodes. It is not a cross-seed or cross-gamma
  measurement, and the write-up says so.
- Both spatial scripts hard-code the per-seed best BDH levels taken from
  `runs/study1/results/table.json`: `sigma_full` at L3, L3, L4 and `sigma_rownorm` at L4, L3, L4.
  Those selections came from `probe_val` during the original run, not from anything computed here.
- Chance rates for the within-radius metrics are computed per row from the true cell's own
  neighborhood size, which shrinks at grid edges and corners. The naive 9/81 and 25/81 would both be
  wrong.

## Provenance

The only edits made when moving these scripts out of the scratchpad and into the repository were to
the file paths: absolute session paths became `HBWM_ROOT` and `HBWM_POSTHOC_OUT` with defaults that
resolve to this repository's worktree layout. No computation, constant, threshold or spec selection
was changed, so the scripts as committed reproduce the numbers reported in RESULTS.md.
