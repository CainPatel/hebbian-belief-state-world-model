# Preregistration (Study 1)

**This file reproduces the preregistered text.** The canonical version is the README as it stood at
commit `e674b1da138f905670dde5571e1a1890b134fe36` ("docs: preregister Study 1 hypotheses and
decision rules", 2026-08-23), which predates every experimental run and is immutable in git history.
**The only changes since are typographic** (em dash removal, and the matching en dashes inside the
steps-since-seen bucket labels). **The rules were fixed before any run and have not been altered:**
no threshold, comparator, feature set or criterion below has been added, removed or moved since that
commit, and every H1 to H4 decision reported in [RESULTS.md](../RESULTS.md) was taken under exactly
these rules. The preregistered text ends at the horizontal rule below; the closing section after
it is post-hoc commentary added on 2026-08-27 and formed no part of the preregistration.

Verify it yourself:

```bash
git show e674b1da138f905670dde5571e1a1890b134fe36:README.md
```

and compare its "Preregistration (Study 1)" section against the text below. The two are identical
once dashes, colons and the parentheses that replaced a pair of em dashes are normalized away.

---

Environment: 9×9 gridworld, 3 distinct-type static objects, agent observes its (x, y) + a 3×3 window, 96 actions/episode, one silent object move in ~50 % of episodes. Models (≈1.58 M params each, same data/steps/optimizer): BDH (γ = 1.0 primary; γ ∈ {0.99, 0.97} secondary), LSTM, RWKV. LR chosen per model from {3e-4, 1e-3, 3e-3} on validation CE at seed 0; then 3 seeds.

Probe target: the true cell of an object that has been seen before and is currently out of view (and not silently moved without re-observation). Linear multinomial probes, L2 chosen on `probe_val`, reported on `probe_test`. Per BDH seed, each feature set reports its best level (chosen on `probe_val`).

Every probe feature is standardized before the linear layer: per-feature mean and standard deviation fitted on `probe_train` (std < 1e-6 → 1), applied identically to BDH features (σ_full, σ_rownorm, x_sparse, resid) and to the LSTM/RWKV state vectors, in training and in the streamed `probe_val`/`probe_test` passes. H4 ranks features by the L2 norm of the probe weights learned on these standardized features. The oracle-memory ceiling (predict the last-seen cell) is 1.0 by construction on this dataset (eligibility excludes stale pairs and static objects never move) and is reported for completeness, not as an informative baseline.

**Decision rules (3 seeds, fixed before any headline run):**

- **H1**: supported iff mean acc(σ_full) exceeds each of {x_sparse, LSTM state, RWKV state} by > 5 points **and** all three paired-by-seed differences are positive. Kill criterion: H1 fails against the LSTM state.
- **H2**: accuracy of the best σ_full probe by steps-since-seen bucket {1-4, 5-8, 9-16, 17-32, 33-64, 65+}. Graceful iff acc(33-64) ≥ 0.5·acc(1-4) and no bucket < 50 % of its predecessor. Reported for each γ arm and both baselines.
- **H3**: on moved + re-observed test episodes: probe p(new cell) vs p(old cell) from re-observation onward; latency = first step with p(new) > p(old). Supported iff latency ≤ 5 steps in ≥ 70 % of such episodes (episodes that never flip count as failures; mean over seeds).
- **H4**: rank σ_full features by probe-weight norm; retrain on top-k for k ∈ {16, 64, 256, 1024, 4096, 16384}; k90 = min k reaching 90 % of full accuracy. Strong: median k90 ≤ 256. Weak: median k90 ≤ 1 % of features. Same procedure on baseline states for relative sparsity.

Anything not listed above is exploratory (notably the `belief()` heatmaps).

---

## What the rules decided

The verdicts, with every supporting number, are in [RESULTS.md](../RESULTS.md#study-1-headline);
the decision rules themselves are implemented as pure functions in `hbwm/probes/decisions.py` and
unit-tested. H1 was **not supported** and its kill criterion **fired**. H2 passes for γ = 1.0 only.
H3 and H4 are **not supported** for any BDH arm.

Study 2 has its own preregistered rules (H5 to H8), which live in
[`docs/superpowers/specs/2026-08-27-hbwm-study2-associative-readout-design.md`](superpowers/specs/2026-08-27-hbwm-study2-associative-readout-design.md).
Study 2 has not been run.
