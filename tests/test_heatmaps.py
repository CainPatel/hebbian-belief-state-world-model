import torch

from hbwm.bdh.core import HBWMConfig, HBWMCore
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.atlas import build_atlas
from hbwm.viz.heatmaps import render_frames

CFG = HBWMConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=34, dropout=0.0, block_size=128)


def test_render_frames(tiny_data, tmp_path):
    torch.manual_seed(0)
    m = HBWMCore(CFG).eval()
    d = EpisodeData(tiny_data.out_dir, "probe_test")
    atlas = build_atlas(m, d, n_episodes=4, top_m=3, batch_eps=4)
    frames = render_frames(m, atlas, d, episode=0, level=1, out_dir=tmp_path, max_steps=3)
    assert len(frames) == 4 and all(p.exists() and p.suffix == ".png" for p in frames)
