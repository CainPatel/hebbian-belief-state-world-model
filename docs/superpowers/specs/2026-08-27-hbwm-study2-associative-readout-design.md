# HBWM Study 2 Detailed Design: Associative Readout of $\sigma$

**Project:** Hebbian Belief-State World Model (HBWM)
**Scope of this document:** Study 2, the pivot mandated by Study 1's preregistered stopping rule. Study 1's H1 was not supported and its kill criterion fired ([RESULTS.md](../../../RESULTS.md)). Study 2 re-asks the belief-readout question with readouts that match how the architecture actually addresses $\sigma$.
**Status:** written 2026-08-27, **before any implementation and before any Study 2 run**. The rules below may be refined up to the preregistration commit and never after it.
**Relationship to Study 1:** [`2026-08-22-hbwm-study1-design.md`](2026-08-22-hbwm-study1-design.md) governs Study 1 and is not amended. This document governs Study 2 and reuses Study 1's data, checkpoints, eligibility rule, standardization, L2 grid, bootstrap CI, and steps-since-seen buckets verbatim, so the two studies' numbers are directly comparable.
**Artifact location:** `runs/` and `data/` are gitignored and physically live in the sibling worktree `.claude/worktrees/study1-impl/`. Every checkpoint, probe cache, and result path in this document is relative to that worktree, for example `.claude/worktrees/study1-impl/runs/study1/bdh_g100_lr0.003/seed0/ckpt.pt`.

---

## 1. Decisions made in this design

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Core question | Format and estimation, not existence | Study 1 tested one readout (a flat linear map). A null there is not a null on $\sigma$'s content |
| 2 | Hypothesis family | Bilinear (low-rank) readouts matching `yKV[h,d] = sum_n q[h,n] sigma[h,n,d]` | The architecture reads $\sigma$ by contracting a sparse positive query against the neuron axis. A readout with that access pattern is the natural test |
| 3 | Honesty about the family | These are **structured linear** readouts, not nonlinear ones | The parameterization is factorized; the score is still linear in $\sigma$. Family 5 separates capacity and nonlinearity from associative structure |
| 4 | Fairness | Every preregistered family runs on the LSTM and RWKV states too | Otherwise Study 2 compares BDH's best readout against the baselines' worst, which is the failure mode that would invalidate the headline |
| 5 | Training-set symmetry | Every family, every model, trains on the **same 24,000-pair** stratified subsample | Removes Study 1's 24,000 vs 61,400 asymmetry from the within-study comparison. Study 1's 61,400 baseline numbers are reported alongside as the cross-study reference |
| 6 | Scope | 9 checkpoints (bdh_g100, lstm, rwkv, seeds 0 to 2), 4 BDH checkpoint-level feature caches | The gamma arms are excluded: Study 1 showed them uniformly weaker and gamma-per-token confounds them |
| 7 | Number of preregistered rules | Exactly four (H5 to H8) | The whole point of a pivot is that it must not become a fishing expedition |

## 2. What Study 1 established

All figures below are quoted from [RESULTS.md](../../../RESULTS.md) and are the only Study 1 numbers this document relies on.

Probe accuracy on `probe_test`, best level per seed, mean over 3 seeds. Chance is 0.011 and the oracle-memory ceiling is 1.000 by construction.

| model | feature | accuracy | #features | n_train | best levels per seed |
|---|---|---|---|---|---|
| bdh_g100 | `sigma_full` | 0.101 | 524,288 | 24,000 | [3, 3, 4] |
| bdh_g100 | `sigma_rownorm` | 0.172 | 8,192 | 61,400 | [4, 3, 4] |
| bdh_g100 | `x_sparse` | 0.062 | 8,192 | 61,400 | [5, 4, 5] |
| bdh_g100 | `resid` | 0.040 | 64 | 61,400 | [5, 3, 5] |
| lstm | `state_vec` | 0.171 | 1,400 | 61,400 | n/a |
| rwkv | `state_vec` | 0.218 | 3,520 | 61,400 | n/a |

- **H1 not supported.** Mean `sigma_full` accuracy beats `x_sparse` by only +0.039 (below the 5-point margin) and loses to the LSTM state by -0.070 and to the RWKV state by -0.117, with every paired-by-seed difference against both baselines negative. The kill criterion ("H1 fails against the LSTM state") fired.
- **H3 not supported.** Fraction of moved-and-re-observed episodes whose belief flips within 5 steps: bdh_g100 0.157, lstm 0.940, rwkv 0.953. The exploratory not-visible-steps-only variant reads 0.130, 0.838, 0.845.
- **H4 not supported for BDH.** Median `k90` is 524,288 (the full feature count) for every BDH arm, against 256 for both baselines.
- **Prediction quality is not the bottleneck.** Test masked CE: bdh_g100 0.0246, lstm 0.0291, rwkv 0.0242.
- **Post-hoc observation (a)** in RESULTS.md: the 8,192-dimensional row-norm view (0.172) beats the 524,288-dimensional flat view (0.101), and one untested candidate explanation is that a flat probe of that width, trained on 24,000 examples, underfits.

## 3. The Study 2 question

Study 1 asked whether a flat linear probe can read object locations out of $\sigma$. It cannot, at 0.101. Study 2 asks the sharper question:

> **Is the belief information present in $\sigma$ but written in an associative, query-addressable format that a flat linear probe of 524,288 free parameters cannot estimate from 24,000 examples?**

The architecture never reads $\sigma$ flatly. Per level and head, `HBWMCore.step` performs

$$y_{\mathrm{KV}}[h, d] \;=\; \sum_{n} q[h, n]\, \sigma[h, n, d], \qquad q = \mathrm{rope}\!\left(\mathrm{relu}(x_t W_{\mathrm{enc}}),\, t\right)$$

with $q$ sparse and positive, and writes $\sigma \leftarrow \gamma\,\sigma + q \otimes x_t$. Content addressed by a query vector along the neuron axis is exactly the structure a flat $\mathrm{vec}(\sigma)$ probe is worst at estimating: the flat probe must learn one free coefficient per $(h, n, d)$ triple, while the associative form needs only the query direction and the value direction. Study 2 preregisters readouts with that access pattern, plus the controls needed to tell "associative structure" apart from "more capacity, fewer parameters, or a nonlinearity".

## 4. Probe families

Notation: $C = 81$ classes (cell ids of a 9x9 grid), $n_h = 4$ heads, $N = 2048$ neurons per head, $D = 64$ value dimensions. $\sigma \in \mathbb{R}^{n_h \times N \times D}$ is one level's state at the feature timestep, per-entry standardized exactly as in Study 1 (mean and std fitted on `probe_train`, std below 1e-6 mapped to 1), then reshaped from the flat 524,288-vector back to $[n_h, N, D]$. Every family has a per-class bias, matching `LinearProbe`.

### 4.1 Family 1, `flat_linear` (control)

$$\mathrm{score}_c = \langle W_c, \sigma \rangle + b_c, \qquad W \in \mathbb{R}^{C \times (n_h N D)}$$

This is Study 1's `sigma_full` probe, refit under Study 2's identical conditions so it is a like-for-like control rather than a quotation. Parameters: $81 \times 524{,}288 \approx 42.5$ M.

### 4.2 Family 2, `query_rank_r` (the hypothesis)

$$\mathrm{score}_c \;=\; \sum_{h=1}^{n_h} \sum_{j=1}^{r} q_{c,h,j}^{\top}\, \sigma_h\, v_{c,h,j} \;+\; b_c, \qquad q \in \mathbb{R}^{C \times n_h \times r \times N},\; v \in \mathbb{R}^{C \times n_h \times r \times D}$$

Each class learns $r$ query directions over neurons and $r$ value directions, per head. Parameters: $C\, n_h\, r\, (N + D) = 81 \cdot 4 \cdot r \cdot 2112 \approx 684$ k per unit of rank. Preregistered grid $r \in \{1, 4, 16\}$, selected on `probe_val`.

This is a **structured (low-rank) linear** readout. Its implied flat weight is $W_c = \bigoplus_h \sum_j q_{c,h,j} v_{c,h,j}^{\top}$, which is a rank-$\le r$ matrix per head. It therefore tests format and estimation efficiency at the same time, and cannot on its own distinguish them; family 5 is the control that separates them.

### 4.3 Family 3, `shared_query_rank_r`

Queries shared across classes, values per class:

$$\mathrm{score}_c = \sum_{h}\sum_{j} q_{h,j}^{\top}\, \sigma_h\, v_{c,h,j} + b_c, \qquad q \in \mathbb{R}^{n_h \times r \times N}$$

Parameters: $n_h r N + C n_h r D = 8192r + 20{,}736r \approx 29$ k per unit of rank, about 24x smaller than family 2. This asks whether a single associative "where is this object" query, read out differently per class, suffices. Same $r$ grid.

### 4.4 Family 4, `derot_flat_linear` and `derot_query_rank_r`

Same as families 1 and 2, after applying the inverse RoPE at the current absolute token position $t$ along $\sigma$'s neuron index. Let $R(t)$ be upstream's rotation, which acts on interleaved neuron pairs $(2j, 2j+1)$ with the paired frequency `Attention.freqs`. Since $\sigma_t = \sum_{s<t} \gamma^{t-1-s} R(s) u_s \otimes x_s$ with $u_s = \mathrm{relu}(x_s W_{\mathrm{enc}})$, applying $R(-t)$ along the neuron axis gives

$$\tilde\sigma_t = R(-t)\,\sigma_t = \sum_{s<t} \gamma^{\,t-1-s}\, R(s-t)\, u_s \otimes x_s,$$

so the stored code becomes position-**relative** rather than absolute-time-rotated. Motivation: RESULTS.md post-hoc observation (a), that row norms, which discard rotation phase entirely, decode at 0.172 against the full flat probe's 0.101.

`derot_flat_linear` is **not** equivalent to `flat_linear`. The transform is orthogonal but depends on the example's own absolute position $t$, so no single fixed $W$ can absorb it; this is the whole reason the family exists. Derotation is applied on the fly when a cached row is loaded, using the pair's absolute token position (`tokenizer.obs_positions(L)[t]`), so no second feature cache is needed.

### 4.5 Family 5, `mlp_rownorm` and `mlp_randproj` (capacity controls)

- `mlp_rownorm`: $8192 \to 512 \to C$ MLP with ReLU, on $\sigma$'s per-neuron row norms, that is on Study 1's `sigma_rownorm` view.
- `mlp_randproj`: a fixed random Gaussian projection of the standardized flat $\sigma$ to 4096 dimensions (seeded per checkpoint, recorded), then $4096 \to 512 \to C$.

These isolate "nonlinearity plus dimensionality reduction" from "associative structure". If a plain MLP on a rotation-invariant 8,192-dimensional summary matches the best bilinear readout, the bilinear result is about capacity and estimation, not about query addressing. Both are reshape-invariant, which matters for the baseline arms (section 5).

### 4.6 Family 6, `synapse_atlas` (exploratory, **not preregistered**)

A readout on the lazy synapse view $\tilde\sigma = \sigma \cdot W_{\mathrm{enc\_v}}$ (`HBWMCore.synapse`), restricted to the concept-atlas row and column index sets that `hbwm/instrument/belief.py`'s `belief_map` already uses: rows in $A_\ell(X_x) \cup A_\ell(Y_y)$, columns in $A_\ell(\mathrm{OBJ}_k)$, plus the transposed term. Reported with a prominent exploratory label, never used to decide anything.

### 4.7 Optimization of the factorized families

Study 1's recipe is held verbatim where it can be: per-feature standardization, Adam lr 1e-3, 20 epochs, batch 512, L2 grid $\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ selected on `probe_val`, reported on `probe_test`, 95% bootstrap CI resampled over test episodes, the same six steps-since-seen buckets. Two additions are forced by the new parameterization and are preregistered here:

1. **L2 applies to the family's own free parameters** ($\|q\|^2 + \|v\|^2$ for factorized families, $\|W\|^2$ for flat and MLP families), matching `train_probes_multi`'s existing penalty on `linear.weight`. The induced penalty on the implied flat $W$ is therefore not identical across families; this is stated in the results table rather than corrected for.
2. **Three random restarts** per factorized family and L2 value, best on `probe_val`. The bilinear objective is nonconvex, unlike Study 1's convex linear probe, and a single unlucky init is a confound that would silently favor the control. Initialization: $q \sim \mathcal{N}(0, 1/N)$, $v \sim \mathcal{N}(0, 1/D)$. Restart count and per-restart `probe_val` accuracy are recorded.

Train accuracy is reported per family alongside test accuracy, so underfitting is visible rather than inferred.

## 5. Fairness: matched families on the baselines

Every preregistered family that can be defined on a baseline state is run on the LSTM and RWKV states. The reshape is preregistered here:

| model | state | matrix form | source of the ordering |
|---|---|---|---|
| BDH | $\sigma$ at the selected level | $[n_h{=}4,\ N{=}2048,\ D{=}64]$ | `features.extract("sigma_full")`, laid out $(h, n, d)$ row-major |
| LSTM | `state_vec`, 1400 | one head, $[1,\ 4,\ 350]$ | `LSTMBaseline.state_vector` concatenates, per layer, $h_i$ then $c_i$ |
| RWKV | `state_vec`, 3520 | one head, $[1,\ 20,\ 176]$ | `RWKVBaseline.state_vector` concatenates, per block, `aa, bb, pp, x_prev_timemix, x_prev_channelmix` |

Each baseline matrix is treated as a single-head "sigma-like" matrix for the rank-$r$ bilinear form. `flat_linear` and the `mlp_*` families apply directly and are reshape-invariant. For families 1, 2, 3 and 5 this is a genuine matched comparison.

**Derotation on baselines is the identity.** Neither baseline carries a rotary phase, so there is nothing to undo; the correct matched definition of `derot_*` on a baseline state is its undecorated counterpart, and the baseline arm reuses that already-fitted probe at no extra cost. This keeps H6 well defined over all five preregistered families.

```mermaid
flowchart TD
    P["Eligible pairs: seen before, currently out of view, not stale<br/>identical 24,000-pair stratified probe_train subsample<br/>full probe_val and probe_test, Study 1 rule unchanged"]

    P --> S1
    P --> S2
    P --> S3

    S1["BDH sigma at the selected level<br/>4 x 2048 x 64"]
    S2["LSTM state_vec 1400<br/>reshaped 1 x 4 x 350"]
    S3["RWKV state_vec 3520<br/>reshaped 1 x 20 x 176"]

    S1 --> FB["families 1 to 5 on sigma<br/>flat, query_rank_r, shared_query_rank_r,<br/>derot variants, mlp_rownorm, mlp_randproj"]
    S2 --> FL["families 1 to 5 on the LSTM state<br/>derot = identity"]
    S3 --> FR["families 1 to 5 on the RWKV state<br/>derot = identity"]

    FB --> H5["H5 format and estimation<br/>best structured sigma readout<br/>minus flat_linear on sigma"]
    FB --> H6
    FL --> H6["H6 HEADLINE, within family<br/>BDH vs LSTM and vs RWKV<br/>kill criterion on the LSTM arm"]
    FR --> H6
    FB --> H7["H7 attribution, reported not gated<br/>mlp_rownorm vs best structured"]
    FB --> H8["H8 belief revision<br/>latency from the first not-visible step<br/>after re-observation"]

    S1 -.-> X["family 6 synapse_atlas<br/>exploratory, BDH only, decides nothing"]
```

## 6. Scope, compute, and cost

**Checkpoints (9).** `bdh_g100`, `lstm`, `rwkv` at lr 0.003, seeds {0, 1, 2}. The gamma arms are excluded from the headline: Study 1 showed them uniformly weaker on every probe hypothesis, and the RESULTS.md caveat that gamma applies per token rather than per environment step ($\gamma^{12}$ per step) means they are confounded as memory-horizon manipulations.

**Levels.** Each BDH seed uses its Study 1 best `sigma_full` level, plus that seed's best `sigma_rownorm` level when it differs, capped at 2 levels per seed. From the table in section 2 (`sigma_full` [3, 3, 4], `sigma_rownorm` [4, 3, 4]):

| seed | `sigma_full` best | `sigma_rownorm` best | Study 2 levels |
|---|---|---|---|
| 0 | L3 | L4 | L3, L4 |
| 1 | L3 | L3 | L3 |
| 2 | L4 | L4 | L4 |

Four BDH (checkpoint, level) feature caches in total.

**Caching.** Features are extracted once per checkpoint-level into an fp16 memmap (24,000 x 524,288 fp16, about 25 GB), and every probe family trains from that one cache, including the derotated families. The cache is deleted after use under the same `try`/`finally` and matrix-level orphan cleanup that Study 1 ended up needing (`hbwm/probes/run.py`, `hbwm/matrix.py`). Baseline caches are 24,000 x 1400 and 24,000 x 3520 fp32, about 134 MB and 338 MB, and stay in RAM. `probe_val` and `probe_test` remain streamed, one recorder pass scoring every candidate probe at once, as in Study 1.

**Cost.** Study 1 measured 5,131 to 5,662 s per BDH probe checkpoint (about 1.4 to 1.6 h) and 50,228.2 s (about 13.95 h) for the whole 15-checkpoint phase, with baselines at 74 to 280 s each. Study 2 runs fewer checkpoints but more probe specs per cache:

| stage | estimate |
|---|---|
| Feature extraction, 4 BDH caches plus 6 baseline states, train + streamed val + streamed test passes | about 11 to 14 h |
| Probe fitting, all families, L2 grid, ranks, restarts | about 3 h |
| **Total** | roughly 1 to 1.5 days of local background wall-clock on MPS |
| Alternative | about 3 to 4 h and 8 to 15 dollars on a rented A100 |

## 7. Preregistered hypotheses and decision rules

Three seeds. Each comparison uses each model's own best level and best hyperparameter (rank, L2, restart) chosen on `probe_val` and reported on `probe_test`. Margins and the paired-positivity convention are Study 1's, unchanged: a 5-point mean margin **and** all three paired-by-seed differences positive.

- **H5 (format and estimation).** Supported iff
  $$\mathrm{mean}\;\mathrm{acc}(\text{best structured } \sigma \text{ readout, families 2 to 4}) - \mathrm{mean}\;\mathrm{acc}(\texttt{flat\_linear on } \sigma) > 0.05$$
  and all three paired-by-seed differences are positive. **Supported means** Study 1's H1 failure was at least partly an artifact of readout format or parameter estimation, not an absence of belief content in $\sigma$.

- **H6 (revised H1, the headline).** For the best **matched** family (families 1 to 5, each defined identically on all three states), supported iff mean acc($\sigma$) exceeds mean acc(LSTM state) by more than 5 points and mean acc(RWKV state) by more than 5 points, with all paired-by-seed differences positive in both comparisons. **Kill criterion: if H6 fails against the LSTM state under matched families, the "sigma as a linearly or bilinearly readable belief state" line is closed. Write up and pivot to the imagination study, or abandon.**

- **H7 (attribution, reported not gated).** Compare `mlp_rownorm` against the best structured $\sigma$ readout. If the MLP is within 2 points of it or better, attribute any gain to capacity and nonlinearity rather than to associative structure. This rule reports an attribution; it gates nothing.

- **H8 (belief revision, revised H3).** Using the best $\sigma$ readout, latency is measured from $t_0(\text{episode}) = \min\{t \ge \texttt{reobserved\_t} : \text{the object is not visible at } t\}$, that is from the first step after re-observation at which the object is **not** visible, rather than from the re-observation step itself at which the answer is inside the agent's 3x3 window. Latency is the first $t \ge t_0$ with $p(\text{new cell}) > p(\text{old cell})$, minus $t_0$; episodes that never flip count as failures. Supported iff latency is 5 steps or less in at least 70% of moved-and-re-observed episodes. RESULTS.md's exploratory not-visible-steps-only column (bdh_g100 0.130, lstm 0.838, rwkv 0.845) is the closest existing reference point, but it filters steps without rebaselining the clock and is therefore not the same statistic.

**Explicitly exploratory and not preregistered:** family 6, per-bucket accuracy curves for the structured families, and any per-head, per-level, or per-rank analysis beyond the `probe_val` selection the rules above require.

## 8. Reuse and new code

Reused unchanged: `hbwm/probes/eligibility.py` (pair sampling, buckets, `h3_pairs`), `hbwm/probes/extract.py` (`iter_features`, `collect_many`, memmap caching and memory hygiene), `hbwm/probes/probe.py` (`feature_stats`, `predict_proba_stream`, `accuracy`, `bootstrap_ci`, `majority_chance`), `hbwm/instrument/{recorder,features,atlas}.py`, `hbwm/probes/evaluate.py`'s aggregation skeleton.

New:

| item | content |
|---|---|
| `hbwm/probes/structured.py` | One `nn.Module` per family, each mapping a standardized flat feature row `[B, F]` to class logits `[B, C]`, so `predict_proba_stream` and the existing joint-L2 training loop work unmodified. Holds the reshape table of section 5, the derotation, the restart logic, and per-family parameter counts |
| `--families` selector in `hbwm/probes/run.py` | Chooses which families to fit for a given checkpoint-level; the default is the Study 2 preregistered set |
| Row context in the runner | The derotated families need each row's absolute token position. `predict_proba_stream` already yields the pair indices, so the runner passes a per-row `t` array alongside the features; families that ignore it are unaffected |
| `h8_latency` in `hbwm/probes/decisions.py` | New function. `h3_latency` is left untouched so Study 1 stays reproducible |
| `h5_decision`, `h6_decision`, `h7_attribution` in `decisions.py` | Pure functions over numbers, in the style of `h1_decision` |
| `experiments/probes/study2.json` | The Study 2 probe preset: family list, rank grid, restart count, checkpoint-level table |

Study 1's standardization, L2 grid, bootstrap CI, and bucket definitions are kept verbatim so cross-study comparison stays legitimate.

## 9. Testing strategy

All tests run on CPU with tiny configs in seconds; TDD throughout, tests written before the code they cover.

| area | tests |
|---|---|
| rank-$r$ expressivity | a `query_rank_r` probe with $r = \min(N, D)$ and unconstrained parameters matches an unconstrained linear probe on a tiny config (same data, same objective, `atol` on the achieved training loss) |
| oracle and null | oracle features (one-hot last-seen cell) give 100% for every family; shuffled labels give chance for every family |
| derotation | derotation is invertible to floating-point tolerance (`derot(derot_inverse(x)) == x`, `atol=1e-5`); derotation of a $\sigma$ built from a single write at step $s$, read at step $t$, has phase $s - t$; `derot_flat_linear` differs from `flat_linear` on data with mixed $t$ |
| baseline reshapes | LSTM 1400 reshapes to `[1, 4, 350]` and RWKV 3520 to `[1, 20, 176]` with the asserted shapes and the orderings that `state_vector` actually emits (asserted against the modules, not hard-coded) |
| matched-family contract | every preregistered family instantiates on all three state shapes; `derot_*` on a baseline is the same object as its undecorated counterpart |
| parameter counts | `flat_linear` 42,467,328; `query_rank_r` $684{,}288 r$; `shared_query_rank_r` $28{,}928 r$, all excluding biases, asserted against the modules |
| decisions | `h5`, `h6`, `h7`, `h8` on hand-built inputs, including the paired-positivity edge cases and the never-flips-counts-as-failure case |
| smoke | end-to-end on the tiny fixture: 8 episodes, a few train steps per model, record states, fit every family, evaluate H5 to H8, all on CPU |

## 10. Milestones and definition of done

| M | content | exit criterion |
|---|---|---|
| M1 | `structured.py` family definitions, parameter-count and expressivity tests, derotation and its tests | rank-$r$ expressivity, derotation, and parameter-count tests green |
| M2 | Baseline reshapes, matched-family contract, `--families` selector, row-context plumbing | matched-family and reshape tests green; a family fits on all three state shapes |
| M3 | `h5` to `h8` decision functions, `experiments/probes/study2.json`, evaluate-side aggregation | decision tests green; smoke test green; `evaluate` produces a Study 2 table from the tiny fixture |
| M4 | README Study 2 preregistration commit, hash recorded in RESULTS.md | commit exists and precedes every Study 2 run |
| M5 | Feature caches and probe fits for all 9 checkpoints, then aggregation, tables, figures | every checkpoint-level in section 6 complete; H5 to H8 decided by the stated rules; parameter counts and `n_train` reported per family |

**Study 2 is done** when M5's exit criterion holds, positive or negative.

## 11. Risks and mitigations

| risk | mitigation |
|---|---|
| The 24,000-example budget still limits `flat_linear`, so H5 partly measures estimation efficiency rather than format | Report parameter counts, `n_train`, and train accuracy per family; read H5 only together with H7, which is the capacity control. State in RESULTS.md that H5 alone cannot separate the two |
| A structured family wins on BDH **and** on both baselines, leaving H6 unchanged | This is a real and reasonably likely outcome and is named in advance. It would mean the associative readout is a better probe generally, not evidence about $\sigma$. H6 is within-family precisely so this outcome is legible rather than confusing |
| Memory and disk pressure repeating Study 1's Jetsam kill | The hygiene already in `hbwm/probes/run.py`: one checkpoint-level at a time, `release_memory` at stage boundaries, memmap flush and close, `[mem]` RSS logging, `try`/`finally` cache deletion, matrix-level orphan-cache cleanup. 25 GB of scratch disk must be free before each cache |
| The study becomes a fishing expedition | Exactly four preregistered rules; families and rank grid fixed before implementation; everything else labeled exploratory and unable to decide anything |
| Nonconvex bilinear objective silently underfits and flatters the convex control | Three restarts per factorized family and L2, best on `probe_val`, with per-restart accuracy recorded; train accuracy reported per family |
| The baseline reshape is architecturally arbitrary (rows are not "neurons" for an LSTM or RWKV the way they are for $\sigma$) and could handicap or flatter the baselines | Named as a limitation. Families 1 and 5 are reshape-invariant, so the H6 headline is checked against at least one reshape-free family before any conclusion is drawn |
| The RWKV state mixes a log-domain component (`pp`) with linear ones in the same reshaped matrix | Study 1's per-feature standardization is applied first and is unchanged, so scales are comparable before any bilinear contraction; the mixing is recorded as a caveat |
| Selecting rank, L2, restart, level and family all on `probe_val` inflates the winner | `probe_val` is 1,000 held-out episodes disjoint from `probe_train` and `probe_test`; every reported number is on `probe_test`; the selection budget per model is stated in RESULTS.md so the reader can discount it |

## 12. Preregistration mechanics

`README.md` gains a Study 2 section carrying H5 to H8 verbatim, committed **before any Study 2 run**, with its commit hash recorded in `RESULTS.md`, exactly as Study 1 did (Study 1's preregistration commit was `e674b1da138f905670dde5571e1a1890b134fe36`). This design document is written before implementation; the rules may be refined up to the preregistration commit and never after it. Any analysis not covered by H5 to H8 is labeled post-hoc in `RESULTS.md`.

## 13. Deferred (explicitly not in this spec)

The imagination and rollout study (Study 2 in SPEC.md's original numbering, which `step()`'s `plasticity` stub already supports); the gamma arms and any per-environment-step decay reparameterization; raising `n_train` toward 61,400 for the flat families; retraining or fine-tuning any model; `share_layers=False`; probes on $\sigma$ deltas or on trajectories of $\sigma$; nonlinear query functions (a learned $q(x_t)$ rather than a learned per-class $q$); attention-style softmax readouts; pixels, doors, keys, RL, planning; CUDA-specific paths or CI; external experiment trackers.

---

## Appendix A: derotation algebra

Upstream's `Attention.rope(phases, v)` rotates interleaved pairs $(v_{2j}, v_{2j+1})$ of the last axis by $2\pi \cdot (\text{phase} \bmod 1)$, with `get_freqs` quantized in pairs so both members of a pair share a frequency. In `step`, `phases = t * freqs` with $t$ the absolute token position, and $k_t = q_t = \mathrm{rope}(u_t, t)$ where $u_t = \mathrm{relu}(x_t W_{\mathrm{enc}})$.

With $R(t)$ the block-diagonal rotation implied by those phases, $R(t)^{-1} = R(-t) = R(t)^{\top}$, and

$$\sigma_t = \sum_{s<t} \gamma^{\,t-1-s}\,\bigl(R(s) u_s\bigr) \otimes x_s \quad \Longrightarrow \quad R(-t)\,\sigma_t = \sum_{s<t} \gamma^{\,t-1-s}\,\bigl(R(s-t) u_s\bigr) \otimes x_s.$$

Two implementation notes. First, `Attention.rope` rotates the **last** axis, while $\sigma$'s neuron axis is the second of three, so the derotation transposes to $[\ldots, D, N]$, applies `rope` with `phases = -t * freqs`, and transposes back. Second, $t$ here is the absolute **token** position of the feature timestep, `tokenizer.obs_positions(L)[t_env]`, not the environment step index; the tests assert this against a single-write $\sigma$ whose expected phase is $s - t$.

## Appendix B: parameter counts

Excluding the $C$ biases, which every family shares.

| family | formula | reference config |
|---|---|---|
| `flat_linear`, `derot_flat_linear` | $C\, n_h N D$ | 42,467,328 |
| `query_rank_r`, `derot_query_rank_r` | $C\, n_h\, r\,(N + D)$ | 684,288 per unit rank: 684,288 / 2,737,152 / 10,948,608 at $r = 1 / 4 / 16$ |
| `shared_query_rank_r` | $n_h r N + C\, n_h r D$ | 28,928 per unit rank: 28,928 / 115,712 / 462,848 at $r = 1 / 4 / 16$ |
| `mlp_rownorm` | $8192 \cdot 512 + 512 \cdot C$ | 4,235,776 |
| `mlp_randproj` | $4096 \cdot 512 + 512 \cdot C$ (projection is fixed, not learned) | 2,138,624 |
| LSTM matched arms | same formulas with $n_h = 1$, $N = 4$, $D = 350$ | `flat_linear` 113,400; `query_rank_1` 28,674 |
| RWKV matched arms | same formulas with $n_h = 1$, $N = 20$, $D = 176$ | `flat_linear` 285,120; `query_rank_1` 15,876 |
