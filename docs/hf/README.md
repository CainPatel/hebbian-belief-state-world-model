# Hugging Face artifact hosting

Cards for two optional Hugging Face repositories. Neither is required to use this project: the code
regenerates the dataset in about 31 seconds, and the GitHub release carries the checkpoints.

| card | what it describes | size |
|---|---|---|
| `MODEL_CARD.md` | the 21 trained checkpoints | 118 MB |
| `DATASET_CARD.md` | the per-episode probe outputs | about 1.7 GB |

The probe outputs are the higher-value artifact: they make every headline number and every post-hoc
claim checkable with numpy alone.

## Uploading

Requires a Hugging Face account and `pip install huggingface_hub`, then `huggingface-cli login`.
`<user>` is the Hugging Face username.

```bash
# Checkpoints, as a model repository.
huggingface-cli repo create hbwm-study1-checkpoints --type model
git clone https://huggingface.co/<user>/hbwm-study1-checkpoints && cd hbwm-study1-checkpoints
cp ../docs/hf/MODEL_CARD.md README.md
tar -xzf hbwm-study1-checkpoints.tar.gz          # the GitHub release asset
git lfs track "*.pt" && git add -A && git commit -m "Study 1 checkpoints" && git push

# Probe outputs, as a dataset repository.
huggingface-cli repo create hbwm-study1-probe-outputs --type dataset
huggingface-cli upload <user>/hbwm-study1-probe-outputs \
  runs/study1 . --repo-type dataset --include "*/probes/*"
```

After uploading, link both from the repository README's "Data and artifacts" section.
