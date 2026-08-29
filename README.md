# Hebbian Belief-State World Model (HBWM): Studies 1 and 2

Does the plastic synapse state $\sigma$ of a BDH ("the Dragon Hatchling") core encode linearly
readable beliefs about a partially observed gridworld? Study 1 trains three parameter-matched
sequence models (BDH, LSTM, RWKV, about 1.58 M parameters each) to predict observations in a 9x9
gridworld, reads their states with standardized linear probes, and decides four hypotheses (H1 to
H4) under rules frozen before any run. **The answer is no: H1 failed, its preregistered kill
criterion fired, and this is published as a preregistered negative result.**

## Hypotheses tested

All four hypotheses, their comparators and their numeric thresholds were fixed at commit
`e674b1d` **before any experimental run**; the full text is in
[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md). Every comparison uses 3 seeds, and each model
reports its own best level chosen on `probe_val` and scored on `probe_test`.

| | what it asks | preregistered bar | verdict |
|---|---|---|---|
| **H1** | Is an out-of-view object's location more linearly readable from $\sigma$ than from BDH's activations alone, or from a parameter-matched LSTM or RWKV state? | mean acc($\sigma$) exceeds *every* comparator by more than 5 points, and all 3 paired-by-seed differences are positive. **Kill criterion:** failing against the LSTM state closes the line of inquiry. | **Not supported.** Kill criterion fired. |
| **H2** | Does that readout decay gracefully as an object stays unseen for longer? | acc(33-64 steps) at least 0.5 times acc(1-4 steps), and no bucket below 50% of its predecessor | **Passes for $\gamma = 1.0$ only**, and weakly. Both baselines fail. |
| **H3** | After an object silently moves and is later seen again, does the belief flip to the new cell? | latency of 5 steps or less in at least 70% of moved-and-re-observed episodes; episodes that never flip count as failures | **Not supported** for any BDH arm. Both baselines pass. Study 2's H8 later found this verdict to be an artifact of the Study 1 clock, which started counting while the object was still visible; see RESULTS.md, "Study 2 headline". |
| **H4** | Is the belief carried by a sparse subset of $\sigma$'s features? | median k90, the smallest top-k reaching 90% of full accuracy, at most 256 (strong) or at most 1% of features (weak) | **Not supported** for BDH. Both baselines are strong-sparse. |

## Result

Top-1 accuracy of a standardized linear probe predicting the true cell (81 classes) of an object the
agent has seen before and cannot currently see. Test split, best level per seed, mean over 3 seeds.
Majority-class chance is **0.011**; the oracle-memory ceiling is **1.000** by construction.

| state read out | probe accuracy | dims |
|---|---|---|
| BDH `sigma_full`, $\gamma = 1.0$ (the hypothesis) | 0.101 ± 0.007 | 524,288 |
| BDH `sigma_rownorm` | 0.172 ± 0.008 | 8,192 |
| BDH `x_sparse` (activations-only ablation) | 0.062 ± 0.009 | 8,192 |
| LSTM state | 0.171 ± 0.006 | 1,400 |
| RWKV state | 0.218 ± 0.007 | 3,520 |

- **H1 not supported, kill criterion fired.** `sigma_full` beats `x_sparse` by only +0.039, short of
  the required 5 points, and loses outright to the LSTM state (-0.070) and the RWKV state (-0.117),
  with every paired-by-seed difference against both baselines negative.
- **H2 passes for $\gamma = 1.0$ only, and weakly.** That arm meets the graceful-decay bar at
  acc(33-64)/acc(1-4) = 0.85, but its curve is uniformly low (0.08 to 0.12 across all six buckets);
  $\gamma = 0.99$, $\gamma = 0.97$, the LSTM and RWKV all fall below the 0.5 ratio bar.
- **H3 not supported for any BDH arm.** Belief flips to the new cell within 5 steps of
  re-observation in 0.157 / 0.300 / 0.352 of episodes against a 0.70 bar; LSTM 0.940, RWKV 0.953.
- **H4 not supported for BDH.** Median k90 is 524,288, the full feature count: none of the six
  tested budgets (16 to 16,384, at most 3.1% of the features) reached 90% of full accuracy, and
  k90 = n_features is a terminal fallback rather than an exhaustive search. Both baselines are
  strong-sparse at median k90 = 256.
- **The model is not broken; the format is the finding.** BDH's test CE is 0.0246 against the
  LSTM's 0.0291 (RWKV 0.0242), so BDH out-predicts the baseline whose state out-probes it. The
  belief drives next-token behavior; what fails is reading it linearly out of $\sigma$.

## Post-hoc findings (exploratory, not preregistered)

Three descriptive follow-ups, run on saved artifacts after the study closed. They **change no
preregistered decision**, and none of them rescues H1. Scripts are in
[analysis/posthoc/](analysis/posthoc/); full numbers and caveats in
[RESULTS.md](RESULTS.md#post-hoc-analyses-exploratory-not-preregistered).

1. **$\sigma$ is structurally sparse and lowish rank.** About 103 of 2,048 neuron rows carry the
   effective mass (participation ratio 0.050), and 10.9 of 64 singular values carry 90% of the
   squared Frobenius mass. This weakens, without eliminating, the "the flat probe simply had too
   many parameters" reading: the effective feature count is nearer 26,000 than 524,288. One
   checkpoint, one level, 64 episodes, so treat the magnitude as indicative.

2. **Probe errors are spatially local.** Within-radius-1 accuracy, against a correctly per-row
   computed chance rate of 0.097: BDH `sigma_full` 0.308, BDH `sigma_rownorm` 0.403, RWKV state
   0.542. Every spec beats both a uniform and a row-shuffled null. The obvious confound is
   controlled: mean distance from prediction to the agent (3.7 to 4.0 cells) tracks the true
   object's distance from the agent (3.81), and predictions land on the agent's own cell only 0.1
   to 1.0% of the time. So all three architectures hold an *approximate* spatial belief that an
   81-way exact-match metric scores as near-total failure.

3. **BDH fades where the baselines corrupt.** Within-radius-1 accuracy by steps since last seen:
   BDH `sigma_rownorm` falls from 0.425 (1-4) to 0.257 (33-64), while RWKV falls from 0.686 to
   0.155 over the same span, so **the ordering inverts at long horizons**. Past 65 steps both
   baselines' expected error *exceeds the uniform null* in all three seeds, meaning confidently
   wrong rather than merely uninformative, while BDH's never crosses its null at any bucket. A blur
   calibration puts BDH at about 2.9 cells, widening to 5 or 6 at the longest gaps, and RWKV at
   about 2.4 cells, widening to about 8 and then unmatchable to any blur scale. Caveats: the 65+
   bucket holds only 662 rows shared across seeds, and for `sigma_full` alone the decay is within
   seed noise.

## Documentation map

- **[docs/EXPLAINER.md](docs/EXPLAINER.md)**: the full account. Architecture, method, every design choice and why, results and honest interpretation.
- **[RESULTS.md](RESULTS.md)**: every number, every table, the caveats, the post-hoc analyses.
- **[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)**: the frozen rules, reproduced from commit `e674b1da138f905670dde5571e1a1890b134fe36`.
- **[SPEC.md](SPEC.md)** and the **[Study 1 design](docs/superpowers/specs/2026-08-22-hbwm-study1-design.md)**: the research proposal, and the buildable spec that governs Study 1.
- **[Study 2 design](docs/superpowers/specs/2026-08-27-hbwm-study2-associative-readout-design.md)** and **[plan](docs/superpowers/plans/2026-08-27-hbwm-study2.md)**: associative readout of $\sigma$, run and complete. H6 was not supported and the preregistered kill criterion fired; see RESULTS.md, "Study 2 headline".
- **[analysis/posthoc/](analysis/posthoc/)**: the three exploratory scripts behind the post-hoc findings.
- **[docs/hf/](docs/hf/)**: model and dataset cards for optional Hugging Face artifact hosting.

## Preregistration (Study 2)

Study 1's H1 was not supported and its kill criterion fired. Study 2 asks whether the belief
information is present in sigma but written in an associative, query-addressable format that a flat
linear probe of 524,288 free parameters cannot estimate from 24,000 examples. Buildable design:
[docs/superpowers/specs/2026-08-27-hbwm-study2-associative-readout-design.md](docs/superpowers/specs/2026-08-27-hbwm-study2-associative-readout-design.md).

Probe families, all on identical eligible pairs, splits, L2 grid, bootstrap CI and steps-since-seen
buckets, and all trained on the same 24,000-pair stratified subsample: `flat_linear` (control),
`query_rank_r`, `shared_query_rank_r`, `derot_flat_linear`, `derot_query_rank_r` with r in {1, 4, 16}
chosen on probe_val, and the capacity control `mlp_state`, joined on BDH by the two reductions
`mlp_rownorm` and `mlp_randproj`. Every family that can be defined on a baseline state is run on the
LSTM state (1,400, reshaped 4 x 350) and the RWKV state (3,520, reshaped 20 x 176); derotation on a
baseline is the identity, and family 5's matched arm is `mlp_state` on all three states, with the two
BDH-only reductions reported as context and feeding H7. Features are standardized per entry on
probe_train exactly as in Study 1; the derotated families derotate the raw state first and standardize
in the derotated frame, with their own statistics, because the two operations do not commute. Every
factorized number is reported with its effective rank fraction r / min(P, Q), clipped at 1.00, and
1.00 marks a saturated arm. `mlp_randproj`'s projection is sparse (64 signed nonzeros per output
dimension, seeded per checkpoint, fixed and not learned) and its construction is recorded.

**Decision rules (3 seeds, fixed before any Study 2 run):**

- **H5 (format and estimation):** supported iff mean acc(best structured sigma readout, families 2 to
  4) minus mean acc(`flat_linear` on sigma) is greater than 5 points and all three paired-by-seed
  differences are positive.
- **H6 (revised H1, the headline):** for the best matched family, supported iff mean acc(sigma)
  exceeds mean acc(LSTM state) and mean acc(RWKV state) by more than 5 points each with all paired
  differences positive. The verdict states which family carried it and whether either baseline arm
  was saturated. A factorized win over a saturated baseline arm is a rank-constraint artifact
  and is flagged as such, so H6 is also read against the reshape-free families. Kill criterion: if H6
  fails against the LSTM state under matched families, the "sigma as a linearly or bilinearly readable
  belief state" line is closed.
- **H7 (attribution, reported not gated):** compare `mlp_rownorm` against the best structured sigma
  readout. If the MLP is within 2 points or better, attribute any gain to capacity and nonlinearity
  rather than to associative structure.
- **H8 (belief revision, revised H3):** latency measured from the first step after re-observation at
  which the object is not visible. Episodes where the object never leaves the window have no such step
  and are excluded from the denominator, not counted as failures; the excluded fraction is reported and
  above 25% the result is flagged low-coverage. Episodes with such a step that never flip are failures.
  Supported iff latency is 5 steps or less in at least 70% of the episodes in the denominator.

**Degeneracy criterion (preregistered, gates H6's selection).** An arm is declared degenerate and
excluded from H6's best-matched-family selection if its probe_train accuracy exceeds 0.95 while its
probe_val accuracy is below twice the majority-class chance rate, at every value of the L2 grid, using
at each L2 the restart that maximizes validation accuracy and applying the test to the seed-mean
accuracies. It applies to every arm of every family and every model. Because an H6 comparison needs
all three arms present, exclusion then operates at the family level: a family is excluded from H6's
best-matched-family selection in full if any one of its three state-arms (BDH, LSTM, RWKV) is
degenerate. Degenerate arms are still fitted and still reported with their parameter count, training
accuracy and validation accuracy, and are labeled degenerate in the results table. This stops a
probe that has merely memorized its training set from deciding a hypothesis. It matters because the
comparison carries a dimensionality asymmetry that
is not specific to any one family: BDH's state is 524,288-dimensional against 1,400 and 3,520 for the
baselines, in Study 2 as in Study 1.

Anything not listed above is exploratory, notably the `synapse_atlas` readout and the descriptive
sigma-structure measurements.

## How to run Study 2

```bash
uv run python -m hbwm.matrix --phase study2 --exp study1 --dry-run
uv run python -m hbwm.matrix --phase study2 --exp study1
uv run python -m hbwm.matrix --phase study2-evaluate --exp study1
```

## Reproduction

```bash
# 0. Environment. pyproject requires Python >= 3.12 and torch >= 2.5; Study 1 ran on
#    Python 3.13.5, torch 2.13.0, numpy 2.5.2 (pinned in uv.lock).
uv sync --extra dev            # add --extra viz for hbwm.viz.heatmaps' animated GIFs
uv run pytest                  # 218 tests, CPU, tiny configs, about 10 s

# 1. Data: 27,000 episodes into data/grid9/{split}.npz (about 31 s, ~14 MB compressed).
#    Every episode is a pure function of its seed, so this regenerates bit-identically.
uv run python -m hbwm.envs.dataset --config experiments/data/grid9.json

# 2. Optional pre-flight: E0 core sanity, the matched baseline widths, the job list.
uv run python -m hbwm.sanity_shakespeare --steps 1000
uv run python -m hbwm.baselines.matching
uv run python -m hbwm.matrix --phase e1 --dry-run

# 3. Training matrix, four phases. Each skips runs whose final.json already exists.
uv run python -m hbwm.matrix --phase e1      # LR sweep, 9 runs
uv run python -m hbwm.matrix --phase e2      # seeds at the best LR, 6 runs
uv run python -m hbwm.matrix --phase e3      # gamma arms, 6 runs
uv run python -m hbwm.matrix --phase probes  # probes for the 15 headline checkpoints

# 4. Hypothesis evaluation (--phase evaluate is a thin wrapper around this).
uv run python -m hbwm.probes.evaluate --root runs --exp study1 --data data/grid9

# 5. Exploratory belief maps. Needs step 3's probes: reads probes/atlas.json, takes
#    the level from probes/done.json.
uv run python -m hbwm.viz.heatmaps --run-dir runs/study1/bdh_g100_lr0.003/seed0 --episode 0

# A single run, or one checkpoint's probes, can also be launched directly:
uv run python -m hbwm.train --config experiments/train/bdh_g100.json --seed 0 --lr 0.003
uv run python -m hbwm.probes.run --run-dir runs/study1/bdh_g100_lr0.003/seed0 --preset study1
```

**Wall-clock, measured from the artifacts, not estimated.** Summing `seconds` over the 21
`final.json` files gives **246,034 s of training (68.3 h)**, and `elapsed_s` over the 15
`probes/done.json` files **50,228 s of probing (14.0 h)**: **82.3 h, about 3.4 days**, so budget
three to four days. Some of that is throttling, not work: runs alternated between lid-closed
dark-wake and awake operation, worth roughly 2x at identical settings.

**Hardware.** Apple Silicon, MPS, fp32 throughout, 128 GB unified memory; `torch.compile` off by
default (a `--compile` flag exists). **Memory warning:** the `sigma_full` probe stage opens one fp16
cache per selected level *concurrently*, about 25 GB each, so the three-level Study 1 default peaks
near **75 GB** of resident scratch, and one probe run was killed by the OS there and re-run after
memory fixes; budget that peak in RAM as well as on disk, or pass a single-level `full_levels`.
CUDA is in-spec but untested; CPU is fully supported and is what the test suite uses.

## Repository layout

| path | what is in it |
|---|---|
| `hbwm/bdh/` | `upstream/` (vendored, never edited, hash-pinned by a test), `core.py` (`HBWMCore`, the decay mask, the recurrent `step()`, the lazy `synapse()` view), `state.py` |
| `hbwm/envs/` | `gridworld.py`, `policies.py`, `tokenizer.py`, `episode.py`, `dataset.py` |
| `hbwm/baselines/` | `lstm.py`, `rwkv.py` (chunked WKV), `matching.py` (the parameter-count solver) |
| `hbwm/instrument/` | `recorder.py` (drives `step()`), `features.py`, `atlas.py`, `belief.py` |
| `hbwm/probes/` | `eligibility.py`, `extract.py`, `probe.py`, `run.py`, `decisions.py` (H1 to H4 as pure functions), `evaluate.py` |
| `hbwm/` | `train.py`, `matrix.py`, `models.py`, `config.py`, `device.py`, `losses.py`, `sanity_shakespeare.py`, `viz/heatmaps.py` (the exploratory belief-map CLI) |
| `experiments/` | `data/grid9.json`, `train/{bdh_g100,bdh_g099,bdh_g097,lstm,rwkv}.json` |
| `analysis/posthoc/`, `notebooks/` | exploratory, read-only on `runs/` and `data/` |
| `tests/` | 218 tests, all CPU, tiny configs, including the equivalence contract and the decision-rule units |

## Data and artifacts

**Committed.** [`results/study1/`](results/study1/) holds the six aggregate JSON files
`hbwm.probes.evaluate` produced, and RESULTS.md's tables are rendered from exactly those files (see
[`results/study1/README.md`](results/study1/README.md)); `docs/figures/` holds the four figures
RESULTS.md references.

**Regenerable, not committed.** `data/grid9/` is about 14 MB and takes about 31 s. `runs/` is about
1.8 GB and takes the 82.3 h above: each run writes `runs/<exp>/<model>_lr<lr>/seed<S>/{config.json,
metrics.jsonl, ckpt.pt, final.json}` (6 MB checkpoints) plus `.../probes/` (about 180 MB of
per-episode outputs per BDH checkpoint), and the aggregation writes `runs/<exp>/results/`. Size is
why the raw checkpoints and per-episode probe outputs are not committed. Both `runs/` and `data/`
are gitignored; here they live in the sibling worktree `.claude/worktrees/study1-impl/`.

**How to verify.** Dataset generation is seed-deterministic and the `.npz` files are byte-stable, so
a fresh `hbwm.envs.dataset` run reproduces them exactly. The reference hashes are:

```
286ec0901fd3c5dd95ef8497bdac5cef53c6de1603470b549a404cac6c11be40  data/grid9/model_train.npz
7901ac78ca3c42360c015cfebf37ec263288b047ea48cf7cd038f3946a79c99e  data/grid9/model_val.npz
67b29c81df2da93e65a7225da5abac8fb8969852bc7316055e4f5885ff9f9ec8  data/grid9/probe_train.npz
89e415f42cff95d24c9d69b43339df59928c5a42237dae336e335ea05e12d429  data/grid9/probe_val.npz
bb3da5fb53272ffd49e651e82989d1d1342c6e2f4280a6e275fcb9364ca09f0e  data/grid9/probe_test.npz
```

Training and probing are not bit-reproducible across devices, so the end-to-end check is that a
re-run of `hbwm.probes.evaluate` lands within seed noise of `results/study1/`.

## Provenance and license

HBWM is released under the MIT License, Copyright 2026 Cain Patel; see [LICENSE](LICENSE), and cite
it with [CITATION.cff](CITATION.cff). The preregistration is commit
`e674b1da138f905670dde5571e1a1890b134fe36`, which predates every experimental run; its text is
reproduced at [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md).

The BDH core is [pathwaycom/bdh](https://github.com/pathwaycom/bdh), vendored **untouched** under
`hbwm/bdh/upstream/` at pinned commit `2b0d7a45b058d4309c84a10e0768d541fe18bdc2` (vendored
2026-08-22, sha256 of `bdh.py` recorded and test-pinned in [`UPSTREAM.md`](hbwm/bdh/upstream/UPSTREAM.md)).
It is **separately MIT licensed, Copyright 2025 Pathway Technology, Inc.**, preserved verbatim at
[`hbwm/bdh/upstream/LICENSE.md`](hbwm/bdh/upstream/LICENSE.md). All HBWM modifications live in
`hbwm/bdh/core.py`, so a `diff` against upstream stays trivial. The BDH reference config is
1,577,216 parameters ($3 n_h D N + 2 V D$ at $n_h = 4$, $D = 64$, $N = 2048$, $V = 34$); the matched
baselines are 1,579,310 (LSTM, +0.13%) and 1,631,168 (RWKV, +3.42%).
