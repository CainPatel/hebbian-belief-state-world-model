# Hebbian Belief-State World Model (HBWM)

**A world model whose belief state is a plastic synapse graph, built on the BDH (Dragon Hatchling) architecture.**

Status: exploratory / open-source research project
Base architecture: [pathwaycom/bdh](https://github.com/pathwaycom/bdh) — Kosowski et al., *The Dragon Hatchling* (arXiv:2509.26507)

---

## 1. Motivation

World models (Dreamer-family RSSMs, JEPA, video-prediction models) compress an agent's history into a latent state used for prediction and planning. That latent is a dense, entangled vector: it works, but nobody can read it. You cannot ask a Dreamer latent "where do you believe the key is?" without training an external probe, and even then the answer is distributed across the whole vector.

BDH offers a different substrate for state. In BDH, inference-time working memory lives entirely in **Hebbian fast weights on the edges of a neuron graph** (the σ state): when two neurons co-fire, their synapse strengthens; unused synapses decay. The BDH paper demonstrated that on language tasks, individual synapses strengthen in response to specific concepts — state is sparse, positive, localized, and inspectable by construction.

**Hypothesis.** If the observation stream of a partially observable environment is fed through a BDH core, the σ state will function as a *readable belief state*: facts about unobserved parts of the world (object locations, latent conditions) will be encoded as identifiable edge weights that persist through occlusion, decay without evidence, and update on contradiction.

If true, this gives world models a property they currently lack: **belief auditing at decision time** ("the model acted this way because edge X was weak") instead of post-hoc explanation.

If false, that is also worth documenting: it would show BDH's monosemanticity on language does not transfer to relational/spatial state, which bounds the architecture's interpretability claims.

---

## 2. Architecture

Four modules. Only one is novel; the design deliberately concentrates all risk in the belief core.

```
observations ──► [1. Encoder] ──► spike pattern x_t ──► [2. BDH Belief Core] ──► graph state (θ fixed, σ plastic)
actions ────────────────────────────────────────────────────┘        │
                                                                     ├──► [3. Prediction head] ──► x̂_{t+1}
                                                                     └──► [4. Decoder / aux heads] ──► ô_t, r̂_t
```

### 2.1 Encoder
- **v0 (gridworld): none.** Discrete observations are tokenized directly — observation symbols and action symbols share one vocabulary, interleaved as a sequence `a_1, o_1, a_2, o_2, …`. This makes the world model literally a language model over an "experience language," which is exactly the regime BDH is validated in.
- **v1+ (pixels): a small frozen or jointly-trained CNN/ViT** whose output is sparsified (top-k or ReLU + normalization) to produce a positive, sparse activation vector over the n neurons. Sparsity and positivity are load-bearing: BDH's interpretability properties are tied to them.

### 2.2 BDH belief core (the contribution)
Direct reuse of BDH-GPU with instrumentation added:
- **Slow weights θ**: learned by backprop during training; encode the environment's dynamics rules ("keys open doors", "objects don't teleport").
- **Fast weights σ**: Hebbian edge state updated during inference; encode the current episode's facts ("*this* key is at (2,3)"). σ is reset between episodes.
- **Instrumentation layer** (new code): hooks that snapshot σ every step, index edges by the neuron clusters they connect, and expose a query API: `belief(concept_a, concept_b) → strength over time`.

Design decisions to resolve empirically (§5):
- σ decay rate: paper default vs. tuned for episode lengths.
- Plasticity during imagination rollouts: full / reduced / frozen (see §2.3).
- Whether layer-sharing (BDH is depth-recurrent) helps or hurts state readability.

### 2.3 Prediction head + imagination
Next-spike-pattern prediction is BDH's native operation — the same forward pass that does next-token prediction. Planning à la Dreamer means rolling the core forward on imagined actions. **Known open problem:** during imagination the Hebbian rule will learn from the model's own hallucinations, potentially reinforcing them (belief drift). Three modes to compare:
1. **Full plasticity** during rollouts (naive baseline; expected to drift),
2. **Frozen σ** (state fixed at rollout start; expected safest, but can't imagine consequences of imagined discoveries),
3. **Scaled-down plasticity** (compromise).

### 2.4 Decoder / auxiliary heads
Linear or shallow heads from graph state to observation reconstruction and reward. Conventional. In v0, reconstruction is next-token cross-entropy and no reward head is needed.

---

## 3. What is different vs. classic world models

| Dimension | Dreamer-style RSSM / JEPA | HBWM (this project) |
|---|---|---|
| Where state lives | Dense latent vector (h, z) | Sparse edge weights σ on a neuron graph |
| State semantics | Entangled; needs trained probes | Hypothesized: localized, queryable associations |
| Object permanence | Engineered (object files, SMC particles) or emergent-and-opaque | Hypothesized native: unrefreshed edges persist until decay/contradiction |
| Context handling | Fixed-size recurrent state or bounded attention window | Linear-time state-space; no fixed context length |
| Online adaptation | None at inference (weights frozen) | Fast weights adapt during deployment by design |
| Uncertainty representation | Implicit in stochastic latent | Open question — σ magnitude may act as confidence, unverified |
| Imagination stability | Well-studied, stable | Open problem (belief drift, §2.3) |
| Maturity / baselines | Years of tuning, strong benchmarks | None; expect worse raw performance initially |

**Closest prior work (and why this is still open):**
- *Fast weights as memory* — Hinton & Plaut 1987; Schmidhuber 1992; modern linear attention. Establishes the substrate; none of it targets readable world-state.
- *Differentiable plasticity* (Miconi et al.) — shows Hebbian coefficients train end-to-end. Mechanism precedent, not applied to world models.
- *Structured World Belief* (Singh et al. 2021) — object-centric belief via SMC particles. Same goal (readable beliefs), heavier machinery, different substrate.
- *Belief-State RWKV* (2026) — interprets a linear-RNN state as a belief state. Philosophically closest neighbor; dense vector rather than plastic graph, so beliefs still need probes rather than being addressable.
- *BDH itself* — validated on language only; community vision port exists (CIFAR-10) but no sequential-decision or belief-state work published.

To the best of current knowledge, **no published work uses Hebbian graph plasticity as the belief state of a world model.** The lane is open, plausibly because the failure modes (§7) are real.

---

## 4. Potential advantages (claims to test, not assume)

1. **Decision-time auditability.** Read the belief that caused an action, rather than explain it post-hoc. Most valuable in safety-relevant autonomy and industrial monitoring.
2. **Free object permanence.** Persistence-by-default of unrefreshed associations, without object-file engineering.
3. **Online world adaptation.** The deployed model updates its beliefs about a *new* environment without retraining — relevant to robotics in unstructured settings.
4. **Unbounded episodes at linear cost.** State-space scaling for long-horizon embodied tasks.
5. **Targeted state editing.** If beliefs are addressable edges, they can be *written*: delete a false belief, inject a known fact. No dense-latent world model can do this surgically. (Stretch goal; would be a striking demo.)

---

## 5. Study plan

Framed as three sequential studies, each gated on the previous one's result. Open-source rigor level: preregister hypotheses in the README, log all runs, report negative results — but no need for 5-seed grids on everything from day one (3 seeds minimum on headline claims).

### Study 1 — Does σ encode readable beliefs? (the gate)
**Environment.** Custom gridworld POMDP, fully controlled: 7×7 to 11×11 grid, 1–4 objects (keys, doors, movable markers), agent observes a 3×3 window. Discrete token observations. Scripted + random exploration policies generate trajectories; no RL needed.

**Model.** BDH-GPU, 1–5M params, trained on next-token prediction over `(action, observation)` sequences. Single consumer GPU / free Colab scale.

**Hypotheses (preregistered):**
- H1: A linear probe on σ decodes the location of an out-of-view object above chance, and above the same probe on (a) activations-only and (b) an LSTM/RWKV baseline's hidden state of matched size.
- H2: Decoding accuracy persists across ≥N steps of occlusion and degrades gracefully (decay curve, not cliff).
- H3: When an object silently moves and the agent re-observes it, the old belief's edge signature measurably weakens and a new one forms (belief revision, visible in σ).
- H4 (strong version): the relevant σ entries are *sparse and localized* — top-k edges suffice for decoding — not merely present in aggregate. This is the difference between "interpretable" and "probe-able," and is the headline claim if it holds.

**Metrics.** Probe accuracy vs. occlusion length; edge-sparsity of decodable signal (k needed for 90% of probe accuracy); belief-revision latency in steps; next-token perplexity vs. LSTM/RWKV baselines (to show interpretability isn't purchased with broken prediction).

**Deliverable.** Repo + writeup + σ-heatmap visualizations ("watch the model's belief about the key decay"). This alone is a complete, publishable-at-workshop-level artifact, positive or negative.

**Kill criterion.** If H1 fails against the LSTM baseline after honest tuning effort, the project's premise is wrong; write it up and stop or pivot.

### Study 2 — Imagination without drift
**Only if Study 1 passes.** Same environment. Add multi-step imagined rollouts under the three plasticity modes (§2.3). Measure: rollout prediction accuracy vs. horizon; belief-state corruption (probe accuracy on σ before vs. after imagination); planning utility via a simple MPC/shooting planner reaching goals that require remembering occluded objects.

**Hypothesis.** Frozen-σ rollouts preserve beliefs; full plasticity degrades them; scaled plasticity sits between — and at least one mode supports planning that beats a memoryless baseline on memory-dependent goals.

### Study 3 — Pixels (stretch)
**Only if Studies 1–2 pass.** MiniGrid pixel renderings or Crafter. Adds the sparse encoder (§2.1). The question shifts to whether readable beliefs survive a learned, continuous frontend — the modality-transfer risk. Success looks like Study 1's probes working at reduced but nontrivial accuracy. This is the step most likely to fail and the least necessary for the project to have been worthwhile.

### Explicitly out of scope
Model-based RL benchmarks at competitive performance; scaling beyond ~50M params; continuous control; any claim of beating Dreamer at anything.

---

## 6. Repo plan

```
hbwm/
  bdh/              # vendored/forked BDH-GPU core (upstream license preserved)
  envs/             # gridworld POMDP + tokenizer; MiniGrid adapter later
  instrument/       # σ snapshotting, edge indexing, belief-query API
  probes/           # linear probes, decay-curve evals, revision detection
  experiments/      # one config file per preregistered run
  notebooks/        # σ visualizations, belief heatmap demos
  RESULTS.md        # running log, including negative results
  SPEC.md           # this document
```

**Milestones.**
1. Fork BDH, reproduce tiny-Shakespeare, add σ instrumentation (weekend-scale).
2. Gridworld env + tokenizer + trained next-token model (2–3 weeks part-time).
3. Study 1 probes and writeup (3–4 weeks part-time). ← first real deliverable
4. Study 2, then optionally Study 3.

**Compute budget.** Studies 1–2: single consumer GPU or free Colab. Study 3: one mid-tier GPU (or a few hundred dollars of cloud).

---

## 7. Risks and honest failure modes

- **Beliefs distributed, not localized** — H1/H4 fail; σ works as memory but not as *readable* memory. Most likely failure. Mitigation: none; this is what the study exists to find out.
- **Spatial ≠ linguistic structure** — BDH's monosemanticity may depend on the discrete compositionality of language. The tokenized gridworld is chosen to be maximally language-like precisely to give the hypothesis its best shot; if it fails *here*, it fails everywhere.
- **Belief drift in imagination** — Study 2's frozen mode is the designed fallback.
- **Baseline unfairness** — an interpretability claim over LSTM/RWKV means matched parameter counts, matched tuning effort, and probes of identical capacity. Reviewers (and honest readers) will check this first.
- **Research-artifact codebase** — BDH's repo is nanoGPT-style, not a library; the graph picture and the GPU tensor implementation don't map one-to-one. Budget real time for reading Sections 2–3 of the paper against the code before writing anything.
- **Scope creep** — the four-module diagram in §2 is the *end state*. Building it before Study 1 produces signal is the canonical way this project dies. v0 is: BDH + tokens + probes.

---

## 8. References

- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz. *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* arXiv:2509.26507 (2025). Code: github.com/pathwaycom/bdh
- Hafner et al. *Dream to Control / DreamerV3* (RSSM world models).
- Singh et al. *Structured World Belief for Reinforcement Learning in POMDP.* ICML 2021.
- Miconi et al. *Differentiable plasticity.* ICML 2018.
- Hinton & Plaut. *Using fast weights to deblur old memories.* CogSci 1987.
- Schmidhuber. *Learning to control fast-weight memories.* Neural Computation 1992.
- *Belief-State RWKV for Reinforcement Learning under Partial Observability.* arXiv (2026).
