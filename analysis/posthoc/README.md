# Post-hoc analysis scripts

Exploratory, descriptive scripts run **after** a preregistered study closed. Their findings are
written up in [RESULTS.md](../../RESULTS.md) under the two "post-hoc" headings and, for the Study 1
set, summarized in [docs/EXPLAINER.md](../../docs/EXPLAINER.md) section 8.

**They decide nothing.** None of them was preregistered, none changes a preregistered decision, and
none reopens a threshold. Study 1's H1 remains not supported with its kill criterion fired, and
Study 2's H6 remains not supported with its kill criterion fired: the "sigma as a linearly or
bilinearly readable belief state" line stays closed. These scripts exist so the post-hoc numbers are
reproducible, not so they can be promoted to findings.

**They are read-only on the model tree.** Nothing here writes into `hbwm/`, retrains a model, or
mutates a checkpoint. The executable tree that produced the published numbers is byte-identical to
what ran. Several scripts do no model inference at all; the rest only drive saved weights forward.
Each writes one JSON, or one block of stdout, and nothing else.

## The Study 1 set

| script | what it measures | model inference? | output |
|---|---|---|---|
| `sigma_structure.py` | How sparse and how low-rank `sigma` is. Row-norm participation ratio, singular-value k90 / k99 / participation ratio, `x_sparse` zero fraction, per-neuron write-mass concentration. | Yes | stdout |
| `spatial_locality.py` | Whether the probe's errors are spatially *local* rather than scattered. Within-radius accuracy against per-row chance, Chebyshev and Manhattan error, expected distance against uniform and row-shuffled nulls. | No | `spatial_locality_results.json` |
| `spatial_locality_buckets.py` | The same, broken down by the six `steps_since_seen` buckets, plus Study 1's `h2_curve` shape test on the graded metric. | No | `spatial_locality_buckets_results.json` |

## The Study 2 set

Written after Study 2 closed, to answer the question the two negatives left open: if the belief is
not linearly readable out of `sigma`, is it there at all, and if so in what form? Run in this order
if you want the narrative to build; each is independent.

| script | what it asks | inference | output JSON (in `results2/`) |
|---|---|---|---|
| `memory_demand_of_objective.py` | Did the training objective ever reward remembering an out-of-view object's absolute position? Data only, no model. | No | stdout |
| `memory_loss_attribution.py` | What share of each model's residual loss sits on the tokens that require that memory? | Yes | stdout |
| `ykv_causal_patch.py` | **Causal.** Replace the associative read `yKV` with another episode's, one level at a time, and see which level's corruption selectively damages memory-dependent predictions. | Yes | `posthoc_ykv_causal.json` |
| `ykv_head_patch.py` | The same intervention per `(level, head)`, plus an additivity test: is the whole-level effect the sum of its heads? | Yes | `posthoc_ykv_head_patch.json` |
| `cancellation_index.py` | The `sum(a)/sum(w)` cancellation index that spec 4.8 measurement 4 defines but the shipped code never reported. Above 1 means writes reinforce, below 1 means they cancel. | Yes | `posthoc_cancellation.json` |
| `cancellation_gamma_arms.py` | The same index for the `bdh_g099` and `bdh_g097` arms, to ask whether lower decay actually reduces interference. | Yes | `posthoc_cancellation_gamma.json` |
| `decodability_vs_time.py` | Does probe accuracy erode as an episode progresses and writes accumulate, controlling for staleness? Pure re-slice of predictions Study 2 already saved. | No | `posthoc_decodability_vs_time.json` |
| `sigma_content.py` | Not "is position in sigma" but "what IS in sigma". SVD of each head's state, with the right singular vectors decoded against the token embeddings. | Yes | `posthoc_sigma_content.json` |
| `frozen_state_rollout.py` | Is the belief stored or continuously refreshed? Freeze the `sigma` update, wholesale and per level, and measure what degrades. | Yes | `posthoc_frozen_rollout.json` |

`ykv_head_patch.py` and `frozen_state_rollout.py` import the token-class labelling, the `_attend`
interceptor and the saturation guard from `ykv_causal_patch.py`, so that one must stay importable.

## Two methodological traps these scripts document

Both would have produced a plausible-looking result rather than an obvious error, so they are
recorded here as well as in the write-up.

- **Saturation.** An intervention severe enough to push overall cross entropy above about 1 nat
  drives every token class to a floor worse than uniform, at which point a contrast between two
  classes carries no information. Patching all six levels at once, and freezing `sigma` wholesale,
  both do this. Those arms are reported as saturation references, and their contrasts are discarded
  rather than presented as nulls. The graded per-level and per-head arms exist to get out of that
  regime.
- **Degenerate output columns.** `lm_head` columns for the ten tokens the model never predicts
  (BOS, PAD, the four actions, and the out-of-range coordinates `X_9`, `X_10`, `Y_9`, `Y_10`) have
  collapsed onto a single shared direction, mean pairwise cosine 0.9999, with inflated norms.
  Ranking any direction against the full vocabulary returns those tokens first, as pure artifact.
  `sigma_content.py` ranks over the trained vocabulary only and reports the degeneracy as a measured
  field.

A third, smaller one: `nn.LayerNorm(elementwise_affine=False)` is scale invariant, so multiplying an
activation by a constant is not an intervention. The causal scripts change content, not scale.

## Running them

The Study 1 set expects the artifact worktree layout and is described by `HBWM_ROOT` and
`HBWM_POSTHOC_OUT`, as before:

```sh
cd .claude/worktrees/study1-impl
uv run python ../../../analysis/posthoc/sigma_structure.py

HBWM_POSTHOC_OUT=/tmp uv run python analysis/posthoc/spatial_locality.py
HBWM_POSTHOC_OUT=/tmp uv run python analysis/posthoc/spatial_locality_buckets.py
```

The Study 2 set resolves the artifact worktree itself and runs from the repository root:

```sh
uv run python analysis/posthoc/memory_demand_of_objective.py     # seconds, no GPU
uv run python analysis/posthoc/decodability_vs_time.py           # under a minute, no GPU
uv run python analysis/posthoc/memory_loss_attribution.py        # minutes
uv run python analysis/posthoc/ykv_causal_patch.py               # about 1 h
uv run python analysis/posthoc/ykv_head_patch.py                 # about 1 h
uv run python analysis/posthoc/cancellation_index.py             # about 30 min
uv run python analysis/posthoc/cancellation_gamma_arms.py        # about 35 min
uv run python analysis/posthoc/sigma_content.py                  # about 15 min
uv run python analysis/posthoc/frozen_state_rollout.py           # about 1 h
```

**Run the model-inference scripts one at a time.** Concurrent MPS jobs contend and one of them was
killed mid-run during this work. On a laptop, also check that the charger is actually negotiating
its full wattage: a 140 W adapter on a 100 W-rated cable delivers 94 W, which is less than sustained
MPS load draws, so the battery drains while plugged in.

## Hard-coded choices worth knowing about

- The Study 1 scripts read one checkpoint, `bdh_g100_lr0.003/seed0`, and hard-code the per-seed best
  levels from `runs/study1/results/table.json`. Those selections came from `probe_val` during the
  original run.
- The Study 2 scripts default to the four checkpoint-levels Study 2 itself measured, seed 0 levels 3
  and 4, seed 1 level 3, seed 2 level 4, with a seeded 1,024-pair subsample, so their numbers slot
  alongside `results2/structure.json`.
- Every script that drives the recorder batches over episodes at `batch_eps=32`. This is not a
  tuning knob: the un-batched form holds roughly 12.6 GB of state and this project lost a checkpoint
  to an OOM kill.
- `cancellation_index.py` recomputes two quantities already published in `results2/structure.json`
  and asserts they reproduce to under 1e-6 relative, which is what proves its sampling, its
  rope-reconstructed query and its accumulator ordering match the pass the shipped code ran.
