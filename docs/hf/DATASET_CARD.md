---
license: mit
tags:
  - interpretability
  - probing
  - world-models
---

# HBWM Study 1 probe outputs

Saved per-pair predictive distributions and metadata from the probe phase of a preregistered
interpretability study. **Every headline number and every post-hoc analysis in the study can be
recomputed from these files with numpy alone, no GPU and no model.**

Code, method and full results: https://github.com/CainPatel/hebbian-belief-state-world-model

## Why this exists

The expensive part of the study is 82 hours of training and probing. These arrays are its output.
Publishing them means a reader can check the reported accuracies, the bootstrap intervals, the
steps-since-seen curves and the exploratory follow-ups (spatial locality of probe errors, the
long-horizon comparison against the baselines) in seconds, rather than trusting the write-up or
spending three days reproducing it.

## Contents

About 1.7 GB across 15 probed checkpoints. Per checkpoint, under `probes/`:

- `<spec>_test.npz`: `probs` float16 `[n, 81]` predictive distributions over grid cells, with
  `label`, `ep` (episode), `t` (step), `obj` (object index), `bucket` (steps-since-seen bucket) and
  `oracle` (last-seen cell). `n` is 41,039 test pairs.
- `<spec>_h3.npz`: the belief-revision readout, `p_old`, `p_new`, `ep`, `t`, `steps_since_reobs`,
  `visible_now`.
- `<spec>.json`: per-probe accuracy, chosen L2, bootstrap interval, per-bucket accuracy and counts,
  and the H4 top-k results.
- `pairs_{train,val,test}.npz`: the sampled probe pairs and their labels.
- `atlas.json`, `done.json`: the concept atlas and the run summary.

Specs are `sigma_full_L<level>`, `sigma_rownorm_L<level>`, `x_sparse_L<level>`, `resid_L<level>` for
BDH and `state_vec` for the baselines.

## Reproducing the analyses

The three exploratory scripts in `analysis/posthoc/` of the GitHub repository read exactly these
files and regenerate the post-hoc tables.

## Limitations

Outputs of one environment configuration and one training protocol. The probes are linear and
standardized on `probe_train` only; the study's negative result is specific to that readout class,
which is the point Study 2 is designed to test.
