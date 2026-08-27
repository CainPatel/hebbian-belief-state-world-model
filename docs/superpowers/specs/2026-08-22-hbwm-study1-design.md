# HBWM: Study 1 Detailed Design

**Project:** Hebbian Belief-State World Model (HBWM)
**Scope of this document:** Study 1 of [SPEC.md](../../../SPEC.md) (§5, "Does σ encode readable beliefs?"), i.e. SPEC.md milestones 1 to 3, plus the minimal interface stub that lets Study 2 reuse the core unchanged.
**Status:** design approved in brainstorming on 2026-08-22; awaiting written-spec review, then implementation planning.
**Relationship to SPEC.md:** SPEC.md is the research proposal and stays as written. This document is the buildable specification for its first study. Where the two differ, this document governs for Study 1.

---

## 1. Decisions made during brainstorming

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Scope | Study 1 only, plus a `plasticity` stub in the recurrent core for Study 2 | Keeps the first deliverable (H1 gate) small; the stub costs ~5 lines and avoids a later core refactor |
| 2 | Primary compute | Local Apple M5 Max / 128 GB, **MPS**, fp32, `torch.compile` off by default | Models are 1 to 5M params; trains locally in ≤ ~1 h/run. CUDA must work but is untested-by-default |
| 3 | σ decay | Hyperparameter `decay_gamma` γ; γ = 1.0 (upstream-faithful) is the preregistered primary arm; γ ∈ {0.99, 0.97} secondary arms | Upstream has no forgetting; H2 needs a controlled knob; γ = 1 keeps the validated model as the control |
| 4 | Agent self-location | Agent's absolute (x, y) is in every observation | Isolates "remember where objects are" from self-localization; gives the hypothesis its best shot (SPEC §7 logic) |
| 5 | Baselines | LSTM (kill-criterion gate) + minimal pure-PyTorch RWKV | LSTM is what the kill criterion names; RWKV is the closest dense-linear-RNN neighbour |
| 6 | σ core implementation | Dual path: upstream parallel form for training; mathematically-equivalent recurrent `step()` for inference/instrumentation; equivalence enforced by tests | No training-speed cost; σ explicit exactly where needed; Study 2 rollouts are `step()` with imagined actions |

## 2. What the upstream code actually is (findings that shaped the design)

Checked against `pathwaycom/bdh@main` (`bdh.py`, 171 lines, MIT; `train.py`, 126 lines):

1. **There is no σ tensor.** Attention is the parallel form `scores = (QR @ KR.mT).tril(-1); y = scores @ V` with `Q = K = relu(x @ encoder)` (sparse, positive, N-dim per head) and `V = x` (dense, D-dim, shared across heads). No softmax. The recurrent equivalent of the state is **N×D per head per layer level**, not n×n.
2. **The n×n "synapse" picture is a view.** σ̃ = σ · encoder_v (N×N, rank ≤ D). Any pair (i, j) is computable from the N×D factor; the full N×N is never needed.
3. **No decay.** The state is a pure sum; RoPE rotates, it does not forget. The paper's graph model has damping; the GPU code does not.
4. **Layer sharing is already on.** One `encoder`, `encoder_v`, `decoder` are reused across the `n_layer` loop. There are therefore `n_layer` distinct σ's per sequence (one per level), because attention runs at every level on a different residual stream.
5. **Upstream training samples random windows** of a byte stream. For σ to be a function of an episode, every sequence must start at episode start.
6. **Upstream `train.py` is CUDA-shaped** (`torch.compile`, fp16 `GradScaler`, autocast only under CUDA).

Reference-config parameter count: `3·nh·D·N + 2·V·D`. Upstream defaults (D=256, nh=4, mult=128 → N=8192, V=256) ≈ 25.3M. Our reference (D=64, nh=4, mult=128 → N=2048, V=34) ≈ 1.58M.

## 3. Architecture

```
GridWorld ─policy─► episode ─tokenizer─► tokens [T]  (+ ground-truth metadata)
                                            │
            ┌───────────────────────────────┼──────────────────────────────┐
            ▼                               ▼                              ▼
     BDH core (hbwm/bdh)             LSTM (hbwm/baselines)         RWKV (hbwm/baselines)
     parallel forward ──train──►      forward ──train──►            forward ──train──►
     recurrent step() ──σ──►          hidden state ──►              wkv state ──►
            │                               │                              │
            └──────────────► SigmaRecorder / feature extractors ◄──────────┘
                                            │
                       ┌────────────────────┼───────────────────┐
                       ▼                    ▼                   ▼
                linear probes          concept atlas +      RESULTS.md tables,
                H1 H2 H3 H4            belief() heatmaps    figures, notebook
```

Only the BDH core + instrumentation is novel. Everything else is deliberately conventional.

## 4. Component designs

### 4.1 BDH belief core: `hbwm/bdh/`

**Files.** `hbwm/bdh/upstream/bdh.py` and `hbwm/bdh/upstream/LICENSE.md` copied verbatim from upstream (provenance: repo URL + commit hash recorded in `hbwm/bdh/upstream/UPSTREAM.md`). `hbwm/bdh/core.py` contains `HBWMCore`, a subclass of upstream `BDH` that overrides `forward` (decay mask, loss mask, internals) and adds the recurrent `step()`. Upstream files are never edited; all changes are in `core.py`, so `diff` against upstream stays trivial.

**Config** (`HBWMConfig`, extends `BDHConfig`):

| field | default | meaning |
|---|---|---|
| `n_layer` | 6 | levels of the shared block |
| `n_embd` (D) | 64 | residual width |
| `n_head` (nh) | 4 | heads |
| `mlp_internal_dim_multiplier` | 128 | N = mult·D/nh = 2048 neurons per head |
| `vocab_size` | 34 | fixed tokenizer vocab (§4.2) |
| `dropout` | 0.1 | as upstream |
| `decay_gamma` | 1.0 | γ; 1.0 reproduces upstream exactly |
| `share_layers` | `True` | `False` allocates per-level `encoder/encoder_v/decoder` (available, not a Study 1 arm) |
| `block_size` | 1164 | max T, used to size the decay-mask buffer |

**Parallel path (training).** Upstream `forward`, with one change: `scores = (QR @ KR.mT) * M` where `M[t, s] = γ^(t−1−s)` for `s < t`, else 0 (a buffer of shape `[block_size, block_size]`, sliced to T). At γ = 1 this is exactly upstream's `tril(diagonal=-1)`. `forward(idx, targets=None, loss_mask=None, return_internals=False)`: `loss_mask` (bool `[B,T]`) restricts CE to observation tokens; `return_internals=True` additionally returns per-level `x_sparse` (`[L,B,nh,T,N]`) for the activations-only probe. Dropout applies only in train mode.

**Recurrent path (inference / instrumentation / Study 2).**

```python
@dataclass
class BDHState:
    sigma: Tensor   # [L, B, nh, N, D], fp32
    t: int          # absolute position of the next token (0-based)

def step(model, tok_t: Tensor[B], state: BDHState,
         plasticity: Literal["full", "frozen", "scaled"] = "full",
         plasticity_scale: float = 1.0) -> tuple[Tensor[B, V], BDHState, Internals]
```

Per level ℓ, with `x_t` the level's residual input (`[B,1,D]`):

1. `x_sparse = relu(x_t @ encoder)` → `[B, nh, N]`
2. `q = k = rope(x_sparse, phase = t · freqs)` (upstream `Attention.rope` with the absolute position t)
3. **read:** `yKV = einsum('bhn,bhnd->bhd', q, σ_ℓ)`: σ_ℓ holds contributions from s < t only, matching `tril(-1)`
4. **write:** `σ_ℓ ← γ · σ_ℓ + α · (k ⊗ x_t)` with α = 1 (`full`), 0 (`frozen`), `plasticity_scale` (`scaled`); `x_t` is broadcast over heads exactly as upstream's `V = x`
5. remaining level math unchanged: `yKV = ln(yKV)`, `y_sparse = relu(yKV @ encoder_v)`, `xy = x_sparse * y_sparse`, `y = ln(xy.reshape(B,1,nh·N) @ decoder)`, `x_{ℓ+1} = ln(x_t + y)`

After L levels: `logits = x @ lm_head`, `state.t += 1`. `Internals` carries per-level `x_sparse` and `yKV`. Study 1 only ever calls `plasticity="full"`; `frozen`/`scaled` exist so Study 2 does not touch the core. There is no `rollout()` in Study 1: `step()`'s signature is the entire stub.

**Synapse view.** `synapse(σ_ℓ, h, rows, cols) = σ_ℓ[h, rows, :] @ encoder_v[h, :, cols]`: lazy, for index sets; full N×N is never materialised by default (a `materialize=True` escape hatch exists for tiny configs / notebooks).

**Equivalence contract (tests, CPU, fp32, tiny config D=16, nh=2, N=64, L=2, T=24, random weights, eval mode):**
- (i) `forward(seq)` logits vs. sequential `step()` logits: `atol=1e-4`, for γ ∈ {1.0, 0.9}.
- (ii) σ after t steps equals the closed form Σ_{s<t} γ^(t−1−s) k_s ⊗ v_s.
- (iii) `frozen` leaves σ bit-identical; `scaled` with scale s produces exactly s × the `full` delta.
- (iv) γ = 1, `share_layers=True`, eval mode → `forward` output bit-identical to upstream `BDH.forward` on the same weights.

**Memory.** Reference config: σ = 6 × 4 × 2048 × 64 × 4 B ≈ 12.6 MB per sequence. Parallel-path attention scores at B=32, T=1164: `32 × 4 × 1164² × 4 B` ≈ 0.7 GB per level (transient). Fine on 128 GB.

**Device.** `select_device()` → `mps` if available, else `cuda`, else `cpu`. fp32 everywhere. `torch.compile` behind a `--compile` flag, default off.

### 4.2 Environment, tokenizer, policies, dataset: `hbwm/envs/`

**World (`gridworld.py`).** `GridWorld(cfg, seed)`, pure numpy, deterministic per seed.

| cfg field | default | notes |
|---|---|---|
| `size` (G) | 9 | interior cells `0..G−1`; 7 ≤ G ≤ 11 supported |
| `n_objects` | 3 | placed on distinct random cells, never on the agent's start cell |
| `n_object_types` (K) | 4 | objects are **distinct** types drawn from K (so "object type k" is unambiguous) |
| `episode_len` (L) | 96 | actions per episode |
| `p_move` | 0.5 | probability of one silent move per episode (H3) |
| `move_window` | `[L/4, 3L/4]` | `t_m ~ Uniform(integer)` |
| `policy_mix` | `{random_walk: ⅓, sweep: ⅓, waypoint: ⅓}` | per-episode policy draw |

Dynamics: agent moves N/E/S/W; moving into the boundary = stay. Objects are static landmarks (no pickup / door semantics in v0); the agent may stand on an object cell. **Silent move:** at `t_m`, pick uniformly among objects *not currently in view*; if none, no move this episode; teleport it to a uniformly random empty cell that is *also* out of view. At most one move per episode. The move is recorded in metadata.

**Observation** at every step: the agent's position and the 3×3 window centred on the agent. Window cells outside the grid read `WALL`; the centre cell reads the object there or `EMPTY`.

**Tokenizer (`tokenizer.py`).** Fixed vocabulary of 34 tokens, independent of G:

| ids | tokens |
|---|---|
| 0 | `BOS` |
| 1 | `PAD` (reserved; unused in Study 1 since all episodes have equal length) |
| 2 to 5 | `A_N, A_E, A_S, A_W` |
| 6 to 16 | `X_0 … X_10` |
| 17 to 27 | `Y_0 … Y_10` |
| 28 | `EMPTY` |
| 29 | `WALL` |
| 30 to 33 | `OBJ_1 … OBJ_4` |

Observation `o_t` = `[X_x, Y_y, c_0 … c_8]` (window row-major): 11 tokens. Sequence = `BOS, o_0, a_1, o_1, …, a_L, o_L`; `T = 1 + 11 + 12·L` = **1164** at L = 96. A loss mask marks observation tokens (`True`) and `BOS`/action tokens (`False`).

**Policies (`policies.py`).** `random_walk` (uniform actions); `sweep` (boustrophedon covering every cell from the start cell, then `random_walk` for the remainder: guarantees every object is seen at least once); `waypoint` (repeatedly pick a uniform random target cell, walk a shortest path to it).

**Dataset (`dataset.py`).** `python -m hbwm.envs.dataset --config experiments/data/grid9.json` writes `data/grid9/{split}.npz` with:
- `tokens` `[n_ep, T]` int16; `loss_mask` `[T]` bool
- `agent_pos` `[n_ep, L+1, 2]`; `obj_pos` `[n_ep, L+1, n_objects, 2]` (per-step, so moves are reflected); `obj_type` `[n_ep, n_objects]`
- `visible` `[n_ep, L+1, n_objects]` bool; `steps_since_seen` `[n_ep, L+1, n_objects]` int (−1 if never seen)
- `moved` `[n_ep]` bool; `move_t` `[n_ep]`; `move_obj` `[n_ep]`; `move_from/move_to` `[n_ep, 2]`; `reobserved_t` `[n_ep]` (first step ≥ `move_t` with the moved object visible, else −1)
- `stale` `[n_ep, L+1, n_objects]` bool: true between a move and its re-observation (label unknowable to the agent)
- `policy` `[n_ep]` int; `seed` `[n_ep]`

Splits by **disjoint seed ranges**: `model_train` 20 000 · `model_val` 1 000 · `probe_train` 3 000 · `probe_val` 1 000 · `probe_test` 2 000 (27 000 episodes ≈ 31 M tokens; ~60 MB as int16). Every training sample is a whole episode beginning at `BOS`.

### 4.3 Trainer and baselines: `hbwm/train.py`, `hbwm/baselines/`

**Trainer.** One loop for all three model families. AdamW (β = 0.9/0.95, `weight_decay` 0.1), cosine schedule with 200 warm-up steps, grad-clip 1.0, fp32, batch 32 whole episodes, `max_steps` 4 000 (≈ 6.4 epochs of `model_train`), masked CE on observation tokens, val every 200 steps on `model_val` (masked CE; also window-cell-only CE), keep best-val checkpoint. Outputs `runs/<exp>/<model>/<seed>/{config.json, metrics.jsonl, ckpt.pt, final.json}`. Entry point: `python -m hbwm.train --config experiments/train/<name>.json --seed S [--lr X]`.

**LSTM (`baselines/lstm.py`).** `Embedding(V, 64) → nn.LSTM(64, H, num_layers=2) → Linear(H, V)`. H is solved numerically so total params are within ±5 % of the BDH reference (expected H ≈ 350). Probed state at step t: `concat(h_1, c_1, h_2, c_2)` after consuming token t.

**RWKV (`baselines/rwkv.py`).** Minimal RWKV-4-style block: time-mix (`R, K, V, O` projections, learnable per-channel `time_decay w` and `time_first u`, token-shift mixing) + channel-mix (`K: C→4C`, `V: 4C→C`, `R: C→C`), pre-LN, 4 blocks; width C solved for param matching (expected C ≈ 176). **WKV is computed in chunks of 64** (within-chunk parallel, log-space `(num, den, max)` carried across chunks) so a forward is not a 1164-step Python loop. Probed state at step t, per block: `[aa, bb, pp, x_prev_timemix, x_prev_channelmix]` concatenated over blocks.

**Fairness protocol (preregistered).** Same data, steps, batch, optimizer, schedule, clip, loss mask for all models. Per model family: LR sweep over {3e-4, 1e-3, 3e-3} at seed 0 on `model_val` CE → best LR → seeds {1, 2} (seed 0 reused). Param counts, chosen LRs, and wall-clock are recorded in `RESULTS.md`.

**BDH arms.** Primary: γ = 1.0. Secondary: γ ∈ {0.99, 0.97}, trained at the γ = 1 LR, 3 seeds each. `share_layers=True` throughout.

**Prediction-quality table.** For every model and seed: test masked CE (all observation tokens) and window-cell CE on `probe_test` episodes (the model never trained on them). This is the "interpretability isn't bought with broken prediction" check.

### 4.4 Instrumentation: `hbwm/instrument/`

**`SigmaRecorder`** (`recorder.py`). Given a checkpoint and a batch of episodes, runs `step()` token-by-token (batched over episodes) and invokes a callback at requested *step indices* (observation boundaries), yielding per-level σ plus `Internals`. Timestep selection: `all`, `every k`, or an explicit list per episode. Nothing is stored unless the callback stores it. Equivalent baseline recorders expose LSTM/RWKV states with the same callback contract.

**Feature extractors** (`features.py`), all per level ℓ, heads concatenated:

| name | dims (reference cfg) | meaning |
|---|---|---|
| `sigma_full` | nh·N·D = 524 288 | flattened σ_ℓ |
| `sigma_rownorm` | nh·N = 8 192 | ‖σ_ℓ[h, i, :]‖₂ per neuron ("synaptic load") |
| `x_sparse` | nh·N = 8 192 | activations-only ablation at step t (from `Internals`) |
| `resid` | D = 64 | residual stream at step t |
| `lstm_state` | 4H ≈ 1 400 | baseline |
| `rwkv_state` | 5·C·blocks ≈ 3 520 | baseline |

Feature timestep (for all models, baselines included) = the last token of observation `o_t` (the model has just read the full observation).

**Concept atlas** (`atlas.py`). For each token v and level ℓ: the mean of `x_sparse` over all positions in a 500-episode sample of `probe_train` where the input token is v; `A_ℓ(v)` = top-32 neurons per head. Saved as JSON per checkpoint.

**Belief query** (`belief.py`). `belief(σ, ℓ, obj=OBJ_k, cell=(x, y))` = Σ over heads of the synapse view summed over `rows ∈ A_ℓ(X_x) ∪ A_ℓ(Y_y), cols ∈ A_ℓ(OBJ_k)` **plus** the transposed term (`rows ∈ A_ℓ(OBJ_k), cols ∈ A_ℓ(X_x) ∪ A_ℓ(Y_y)`). Returns a scalar; `belief_map(σ, ℓ, obj)` returns the G×G grid. This is the exploratory, human-readable view that drives the heatmaps; it is **not** a preregistered claim and carries a prominent docstring saying so.

### 4.5 Probes and hypothesis evaluators: `hbwm/probes/`

**Labels and eligibility.** Target for (episode, t, object k) = the object's true cell id ∈ {0 … G²−1} at step t. **Eligible** iff `steps_since_seen ≥ 1` (seen before, out of view now) **and not `stale`**. Per episode and object: up to 8 eligible pairs sampled stratified over `steps_since_seen` buckets {1 to 4, 5 to 8, 9 to 16, 17 to 32, 33 to 64, 65+}.

**Probe model.** Linear multinomial logistic regression (no hidden layer), one per (model checkpoint, feature set, level). Adam lr 1e-3, 20 epochs, batch 512, L2 ∈ {1e-4, 1e-3, 1e-2, 1e-1} chosen on `probe_val`; trained on `probe_train`; reported on `probe_test`. Feature sets ≤ 16 k dims use all eligible `probe_train` examples, cached in RAM (fp32). `sigma_full` probes train on a stratified subsample of 24 000 `probe_train` examples, cached as fp16 on disk one level at a time (≈ 25 GB per level, deleted after use); their `probe_val`/`probe_test` evaluation is a single streaming pass of the recorder in which all candidate probes (the L2 grid, and the H4 top-k variants) are evaluated at once, so val/test features are never cached.

**Level selection.** `sigma_rownorm`, `x_sparse`, `resid` are probed at all 6 levels. `sigma_full` is probed at the two levels with the highest `sigma_rownorm` `probe_val` accuracy **plus** the last level (up to 3 levels). "Best level" for any feature set is chosen on `probe_val` and reported on `probe_test`.

**Reported per probe:** top-1 accuracy, majority-class chance, number of features, 95 % bootstrap CI over test episodes, and the **oracle-memory ceiling** (predict the last-seen cell; 100 % on unmoved objects).

**Preregistered decision rules** (3 seeds; each comparison uses each model's own best level):

- **H1**: supported iff `mean_seeds acc(sigma_full) − mean_seeds acc(C) > 5 pts` for every comparator `C ∈ {x_sparse, lstm_state, rwkv_state}` **and** all three paired-by-seed differences are positive. Kill criterion: H1 fails against `lstm_state` → write up and stop/pivot (SPEC §5).
- **H2**: accuracy of the best `sigma_full` probe by `steps_since_seen` bucket. "Graceful" iff acc(33 to 64) ≥ 0.5 · acc(1 to 4) **and** no bucket is < 50 % of its predecessor. Reported for γ ∈ {1.0, 0.99, 0.97} and for both baselines.
- **H3**: on `probe_test` episodes with `moved` and `reobserved_t ≥ 0`, for the moved object: `p(new cell)` and `p(old cell)` from the best `sigma_full` probe vs. steps since `reobserved_t`. Latency = first step at which `p(new) > p(old)`. Supported iff latency ≤ 5 steps in ≥ 70 % of such episodes. `belief()` curves for old vs. new cell are plotted alongside (exploratory).
- **H4**: on the best `sigma_full` probe: rank features by ‖w_f‖₂ across classes; for k ∈ {16, 64, 256, 1 024, 4 096, 16 384, all} retrain the probe from scratch on the top-k features (same recipe); `k90` = min k with acc ≥ 0.9 · acc(all). Localisation = number of distinct neurons (rows) touched by the top-`k90` features. **Strong H4:** `k90 ≤ 256`. **Weak H4:** `k90 ≤ 1 %` of features. The same procedure is run on `lstm_state` and `rwkv_state` to report relative sparsity (`k90 / n_features`).

**Entry points.** `python -m hbwm.probes.run --ckpt … --features sigma_rownorm --levels all`; `python -m hbwm.probes.evaluate --exp <name>` aggregates across seeds and writes `runs/<exp>/results/{h1,h2,h3,h4}.json` + figures + a Markdown table block for `RESULTS.md`.

### 4.6 Visualisation deliverables: `notebooks/`

`belief_heatmaps.ipynb`: load a checkpoint and one `probe_test` episode, run `SigmaRecorder` at every step, draw per-step G×G `belief_map` for each object next to the true map and the agent's window, and export a PNG frame sequence (GIF via `imageio` if installed). Also `sigma_decay.ipynb`: H2 curves and the top-`k90` edge set rendered on the neuron graph (rows vs. columns). Both use only `runs/` artefacts.

## 5. Experiment matrix and compute

| exp | runs | purpose |
|---|---|---|
| E0 sanity | 1 | upstream `bdh.py` config scaled to D=128 on tiny-Shakespeare, 1 000 steps on MPS: loss decreases monotonically and samples are non-degenerate. No hard target (config differs from upstream's) |
| E1 LR sweep | 9 | {BDH γ=1, LSTM, RWKV} × {3e-4, 1e-3, 3e-3}, seed 0 |
| E2 seeds | 6 | {BDH γ=1, LSTM, RWKV} × seeds {1, 2} at best LR |
| E3 γ arms | 6 | BDH γ ∈ {0.99, 0.97} × seeds {0, 1, 2} |
| **total training runs** | **22** | |

Probing: per BDH checkpoint (9 runs): `sigma_rownorm` ×6 levels, `x_sparse` ×6, `resid` ×6, `sigma_full` ×3, each × 4 L2 values; per baseline checkpoint (6 runs): 1 feature set × 4 L2. H4 adds 7 retrains per BDH seed for γ=1 and 7 per baseline seed.

Estimate on MPS: BDH ≈ 0.5 to 1 h/run; LSTM/RWKV similar; feature extraction ≈ minutes per checkpoint; `sigma_full` probes ≈ 10 to 20 min each. Whole matrix ≈ 1 to 2 days of background wall-clock.

**Preregistration mechanics.** `README.md` contains §4.5's hypotheses and decision rules verbatim and is committed **before** any E1 to E3 run starts; `RESULTS.md` records that commit hash. Any post-hoc analysis is labelled as such.

## 6. Repository layout, tooling, configuration

```
.
├── SPEC.md                      # research proposal (unchanged)
├── README.md                    # preregistration + how to run
├── RESULTS.md                   # running log incl. negative results
├── pyproject.toml               # uv-managed; requires-python >=3.12
├── docs/superpowers/specs/      # this document (+ later Study 2/3 specs)
├── experiments/
│   ├── data/grid9.json
│   ├── train/{bdh_g100,bdh_g099,bdh_g097,lstm,rwkv}.json
│   └── probes/{study1.json}
├── hbwm/
│   ├── bdh/{upstream/{bdh.py,LICENSE.md,UPSTREAM.md}, core.py, state.py}
│   ├── envs/{gridworld.py, tokenizer.py, policies.py, dataset.py}
│   ├── baselines/{lstm.py, rwkv.py}
│   ├── instrument/{recorder.py, features.py, atlas.py, belief.py}
│   ├── probes/{probe.py, eligibility.py, run.py, evaluate.py, h1.py, h2.py, h3.py, h4.py}
│   ├── train.py
│   ├── config.py                # dataclass ⇄ JSON loader
│   └── device.py
├── tests/
├── notebooks/
├── data/                        # gitignored
└── runs/                        # gitignored
```

- **Dependencies:** `torch`, `numpy`, `matplotlib`, `tqdm`; dev: `pytest`, `ruff`. Optional: `imageio`, `jupyter`.
- **Configs:** plain `@dataclass` per component; `hbwm.config.load(path, cls)` reads JSON; every run writes its resolved config back. No Hydra.
- **Logging:** `metrics.jsonl` (one JSON object per eval step), `final.json`, PNG figures. No external tracking services.
- **Git:** repo initialised 2026-08-22; `.gitignore` covers `data/`, `runs/`, `.venv/`, `__pycache__/`, `*.npz`, `*.pt`. Upstream license preserved under `hbwm/bdh/upstream/`.

## 7. Testing strategy

All unit tests run on CPU with tiny configs in seconds; TDD throughout (tests written before the code they cover).

| area | tests |
|---|---|
| core | equivalence contract (i) to (iv) of §4.1; decay-mask values; `share_layers=False` allocates distinct parameters; `synapse()` equals explicit `σ @ encoder_v` on a tiny config |
| env | seed determinism; window contents at corners/edges; boundary bump = stay; move invariants (object out of view before and after; at most one move); `stale`/`reobserved_t` bookkeeping; policy coverage (fraction of `sweep` episodes where every object is seen = 1.0) |
| tokenizer | round-trip; vocab size 34; loss mask is `False` for `BOS` and all 96 action tokens and `True` for all 97 × 11 = 1 067 observation tokens |
| dataset | shapes; disjoint seeds across splits; `steps_since_seen` consistent with `visible` |
| trainer | tiny-batch overfit (loss strictly decreases over 30 steps) for all three models; checkpoint round-trip; LR sweep selects the argmin |
| baselines | param-matching within ±5 %; RWKV chunked WKV == naive sequential WKV (`atol=1e-5`) incl. across chunk boundaries; LSTM state extractor shape |
| instrument | recorder yields σ at exactly the requested steps; extractor dims match the table in §4.4; `belief_map` shape G×G |
| probes | oracle features (one-hot last-seen cell) → 100 %; shuffled labels → chance; eligibility mask matches a hand-computed example; top-k accuracy is non-decreasing in k up to noise (asserted on synthetic data); H1 to H4 decision functions on hand-built inputs |
| smoke | end-to-end: 8 episodes → 20 train steps per model → record σ → fit a probe → evaluate H1 function, all on CPU |

## 8. Milestones and definition of done

| M | content | exit criterion |
|---|---|---|
| M1 | vendored upstream, `core.py` parallel+recurrent paths, `step()` with plasticity stub, equivalence tests, E0 sanity run on MPS | all §4.1 tests green; E0 loss curve logged |
| M2 | env, tokenizer, policies, dataset generator; BDH trains on `grid9` | env/tokenizer/dataset tests green; one BDH run to 4 000 steps with val CE logged |
| M3 | LSTM + RWKV baselines, param matching, fairness sweep tooling | baseline tests green; E1 runnable end-to-end |
| M4 | recorder, feature extractors, atlas, `belief()`, probes, H1 to H4 evaluators, notebooks skeleton | probe tests + smoke test green; `evaluate` produces tables from a tiny run |
| M5 | README preregistration commit → E1 to E3 → probes → `RESULTS.md`, figures, heatmap notebook | every run in §5 complete; H1 to H4 decided by the stated rules; perplexity table; notebook renders |

**Study 1 is done** when M5's exit criterion holds: positive or negative result.

## 9. Risks and mitigations

| risk | mitigation |
|---|---|
| MPS missing an op / numerical quirk | fp32 only; `PYTORCH_ENABLE_MPS_FALLBACK=1` documented; all tests also pass on CPU |
| `sigma_full` probes overfit (524 k features, ~24 k examples) | L2 grid on `probe_val`; report feature counts; `sigma_rownorm` + top-k give the lower-dimensional story; H4 is the sparsity check |
| Feature-count mismatch between BDH and baselines | Linear probe everywhere, regularisation chosen per feature set, feature counts and CIs reported; relative `k90` reported for all |
| RWKV training speed on MPS | chunked WKV (§4.3); if still > 2 h/run, reduce chunk size tests to CPU and run RWKV last |
| LSTM slow on MPS | `nn.LSTM` is supported on MPS; if > 2 h/run, run on CPU with the same config (results are device-independent up to fp32 noise) |
| σ memory / disk in the recorder | recorder never stores unless asked; `sigma_full` train cache is one level at a time as fp16 (≈ 25 GB scratch disk required); val/test streamed |
| Upstream repo changes | vendored at a pinned commit, hash recorded |

## 10. Deferred (explicitly not in this spec)

Study 2 rollouts/imagination (only `step()`'s `plasticity` stub), Study 3 / pixels / encoder, door-key semantics, `share_layers=False` arm, agent-position-hidden variant, RL or planning, CUDA-specific paths or CI, external experiment trackers.

---

## Appendix A: Parallel ⇄ recurrent equivalence

Per level and head, with `k_s = q_s = rope(relu(x_s @ enc), s)` and `v_s = x_s`:

- Parallel: `y_t = Σ_{s<t} M[t,s] (q_t · k_s) v_s`, `M[t,s] = γ^(t−1−s)`.
- Recurrent: `σ_t = Σ_{s<t} γ^(t−1−s) k_s ⊗ v_s`, so `σ_{t+1} = γ σ_t + k_t ⊗ v_t` and `y_t = q_tᵀ σ_t`.
- RoPE: upstream rotates Q (and K = Q) by phase `t · freqs` at absolute position t; the recurrent path passes the same absolute t, so `q_t · k_s` depends on `t − s` exactly as in the parallel form.
- Read-before-write at each step reproduces `tril(diagonal=-1)`.

## Appendix B: Parameter counts

- BDH (shared): `3·nh·D·N + V·D (embed) + D·V (lm_head)`; reference = 3·4·64·2048 + 2·34·64 = **1 577 216**.
- BDH (`share_layers=False`): `L·3·nh·D·N + 2·V·D`.
- LSTM: `V·E + 4(E·H + H² + 2H) + 4(2H² + 2H) + H·V + V`, E = 64; solve H for ±5 % of reference.
- RWKV (4 blocks): per block ≈ `13C² + O(C)` (mix vectors, `w`, `u`, LayerNorms); plus `2·V·C`; solve C for ±5 %.
