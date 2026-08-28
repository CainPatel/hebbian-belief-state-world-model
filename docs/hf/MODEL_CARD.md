---
license: mit
library_name: pytorch
tags:
  - interpretability
  - mechanistic-interpretability
  - world-models
  - hebbian-learning
  - associative-memory
---

# HBWM Study 1 checkpoints (BDH, LSTM, RWKV on a 9x9 gridworld)

**These are not useful language models.** They are 21 research checkpoints of about 1.58 M
parameters each, trained on a synthetic 34-token vocabulary describing a 9x9 gridworld. They exist
so that a preregistered interpretability study can be reproduced or extended without repeating 68
hours of training.

Code, method and full results: https://github.com/CainPatel/hebbian-belief-state-world-model

## What the study found

A standardized linear probe reads an out-of-view object's cell out of BDH's Hebbian synapse state
sigma at 0.101, below a parameter-matched LSTM state's 0.171 and RWKV state's 0.218 (chance 0.011,
oracle ceiling 1.000). The preregistered hypothesis H1 failed and its kill criterion fired. BDH
still out-predicts the LSTM on test cross-entropy (0.0246 against 0.0291), so the null concerns
readout format rather than model quality. Rules were frozen at commit `e674b1d` before any run.

## Contents

21 checkpoints: `bdh_g100` (gamma = 1.0), `bdh_g099`, `bdh_g097`, `lstm` and `rwkv`, at the swept
best learning rate of 3e-3, seeds 0 to 2, plus the 9 learning-rate-sweep runs at seed 0. Each ships
with its `config.json` and `final.json`. `MANIFEST.json` lists model, seed, learning rate, best
validation cross-entropy, best step, parameter count and sha256 for every one.

## Usage

```python
from hbwm.train import load_checkpoint          # from the GitHub repo
model, cfg, meta = load_checkpoint("bdh_g100_lr0.003/seed0/ckpt.pt", device="cpu")
```

These are `torch.save` pickles holding a `model_state` and a config dict, so the loader uses
`weights_only=False`. Load only artifacts you trust, as with any pickle.

## Architecture and provenance

BDH ("the Dragon Hatchling") is vendored unmodified from `pathwaycom/bdh` at a pinned commit and is
separately MIT licensed (Pathway Technology, Inc.); see `hbwm/bdh/upstream/UPSTREAM.md`. `HBWMCore`
subclasses it to add a gamma decay term and a mathematically equivalent recurrent `step()` that
materializes sigma, with equivalence enforced by tests. Reference configuration: 6 shared levels,
d_model 64, 4 heads, 2048 neurons per head, 34-token vocabulary, 1,577,216 parameters.

## Limitations

Trained on one synthetic environment with one seed family. The models are not evaluated on, and are
not useful for, natural language or any downstream task. The study's negative result is about
linear readability of a specific state, not about BDH's capability.
