# Hebbian Belief-State World Model (HBWM) — Study 1

A world model whose belief state is the plastic synapse state σ of a BDH (Dragon Hatchling) core.
Proposal: [SPEC.md](SPEC.md). Buildable design: [docs/superpowers/specs/2026-08-22-hbwm-study1-design.md](docs/superpowers/specs/2026-08-22-hbwm-study1-design.md). Results log: [RESULTS.md](RESULTS.md).

## Preregistration (Study 1)

Environment: 9×9 gridworld, 3 distinct-type static objects, agent observes its (x, y) + a 3×3 window, 96 actions/episode, one silent object move in ~50 % of episodes. Models (≈1.58 M params each, same data/steps/optimizer): BDH (γ = 1.0 primary; γ ∈ {0.99, 0.97} secondary), LSTM, RWKV. LR chosen per model from {3e-4, 1e-3, 3e-3} on validation CE at seed 0; then 3 seeds.

Probe target: the true cell of an object that has been seen before and is currently out of view (and not silently moved without re-observation). Linear multinomial probes, L2 chosen on `probe_val`, reported on `probe_test`. Per BDH seed, each feature set reports its best level (chosen on `probe_val`).

Every probe feature is standardized before the linear layer — per-feature mean and standard deviation fitted on `probe_train` (std < 1e-6 → 1), applied identically to BDH features (σ_full, σ_rownorm, x_sparse, resid) and to the LSTM/RWKV state vectors, in training and in the streamed `probe_val`/`probe_test` passes. H4 ranks features by the L2 norm of the probe weights learned on these standardized features. The oracle-memory ceiling (predict the last-seen cell) is 1.0 by construction on this dataset — eligibility excludes stale pairs and static objects never move — and is reported for completeness, not as an informative baseline.

**Decision rules (3 seeds, fixed before any headline run):**

- **H1** — supported iff mean acc(σ_full) exceeds each of {x_sparse, LSTM state, RWKV state} by > 5 points **and** all three paired-by-seed differences are positive. Kill criterion: H1 fails against the LSTM state.
- **H2** — accuracy of the best σ_full probe by steps-since-seen bucket {1–4, 5–8, 9–16, 17–32, 33–64, 65+}. Graceful iff acc(33–64) ≥ 0.5·acc(1–4) and no bucket < 50 % of its predecessor. Reported for each γ arm and both baselines.
- **H3** — on moved + re-observed test episodes: probe p(new cell) vs p(old cell) from re-observation onward; latency = first step with p(new) > p(old). Supported iff latency ≤ 5 steps in ≥ 70 % of such episodes (episodes that never flip count as failures; mean over seeds).
- **H4** — rank σ_full features by probe-weight norm; retrain on top-k for k ∈ {16, 64, 256, 1024, 4096, 16384}; k90 = min k reaching 90 % of full accuracy. Strong: median k90 ≤ 256. Weak: median k90 ≤ 1 % of features. Same procedure on baseline states for relative sparsity.

Anything not listed above is exploratory (notably the `belief()` heatmaps).

## How to run

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m hbwm.envs.dataset --config experiments/data/grid9.json
uv run python -m hbwm.matrix --phase e1      # LR sweep (9 runs)
uv run python -m hbwm.matrix --phase e2      # seeds at best LR (6 runs)
uv run python -m hbwm.matrix --phase e3      # gamma arms (6 runs)
uv run python -m hbwm.matrix --phase probes  # probes for the 15 headline checkpoints
uv run python -m hbwm.probes.evaluate --root runs --exp study1 --data data/grid9
uv run python -m hbwm.viz.heatmaps --run-dir runs/study1/bdh_g100_lr<best>/seed0 --episode 0
```

Upstream BDH is vendored under `hbwm/bdh/upstream/` (MIT, pinned commit in `UPSTREAM.md`) and never edited.
