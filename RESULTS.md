# HBWM Results Log

## Dataset grid9

Generated with `uv run python -m hbwm.envs.dataset --config experiments/data/grid9.json` (wall-clock ~31s for 27,000 episodes; grid=9x9, 3 objects, episode_len=96, T=1164 tokens/episode).

Sanity check (`probe_test` split, n=2000):
```
uv run python -c "from hbwm.envs.dataset import EpisodeData as E; d=E('data/grid9','probe_test'); import numpy as np; print(d.n, d.T, d.moved.mean(), (d.reobserved_t>=0).mean(), d.visible.mean())"
2000 1164 0.5205 0.311 0.09749828178694159
```

- Fraction of episodes with an object moved: **0.5205** (expected ~0.45-0.5)
- Fraction of episodes where the moved object is re-observed before the episode ends: **0.311** (expected ~0.3-0.45)
- Fraction of (t, obj) pairs where the object is visible: **0.0975** (expected ~0.1-0.2)
