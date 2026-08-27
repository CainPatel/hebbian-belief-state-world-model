# Study 1 aggregate results

These six JSON files are the aggregated outputs of the probe phase, exactly as
`hbwm.probes.evaluate` wrote them. **RESULTS.md's Study 1 tables are rendered from these files and
nothing else**, so every headline number in this repository can be checked here without rerunning
anything.

They were produced by:

```bash
uv run python -m hbwm.probes.evaluate --root runs --exp study1 --data data/grid9
```

which reads the 15 headline checkpoints' `runs/study1/*/seed*/probes/` outputs, applies the
preregistered decision rules in `hbwm/probes/decisions.py`, and writes
`runs/study1/results/`. The files here are byte-identical copies of that directory's JSON outputs.

| file | contents |
|---|---|
| `table.json` | probe accuracy per model and feature set: `per_seed`, `mean`, `std`, selected `levels` and `specs`, 95% bootstrap `ci95` per seed, `n_train`, `n_features`, `chance`, `ceiling` |
| `h1.json` | the H1 verdict: `bdh_mean`, the 0.05 `margin`, and per comparator (`x_sparse`, `lstm`, `rwkv`) the `mean`, `mean_diff`, `paired_diffs` and `passes` |
| `h2.json` | per model: accuracy `values` by steps-since-seen bucket, `ratio_33_64_over_1_4`, `graceful`, `spec_per_seed`, `bucket_n` |
| `h3.json` | per model: `per_seed` belief-revision latencies, `mean_frac_le5`, `supported`, and the exploratory `mean_frac_le5_not_visible` variant |
| `h4.json` | per model: `per_seed` accuracy-by-k curves, `median_k90`, `n_features`, and the `strong` / `weak` sparsity verdicts |
| `perplexity.json` | the prediction-quality check: per model `n_params`, `lr`, `val_ce_mean`, `val_ce_std`, `test_ce_mean`, `test_ce_window_mean` |

The same run also writes `results.md` (the rendered tables, pasted verbatim into RESULTS.md) and
`h2_curves.png` / `h4_curves.png` (copied into `docs/figures/`).

## What is not here, and why

Raw checkpoints (`ckpt.pt`, 6 MB each) and per-episode probe outputs (the `.npz` files under
`runs/study1/*/seed*/probes/`, roughly 180 MB per BDH checkpoint, about 1.8 GB in total) are **not
committed, because of their size**. They are regenerable: see the Reproduction section of the
top-level [README.md](../../README.md). Training and probing are not bit-reproducible across
devices, so the end-to-end check is that a re-run of `hbwm.probes.evaluate` lands within seed noise
of these files.
