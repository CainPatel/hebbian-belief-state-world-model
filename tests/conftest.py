import pytest

from hbwm.envs.dataset import DataConfig, generate
from hbwm.envs.gridworld import GridConfig

TINY_GRID = dict(size=5, n_objects=2, n_object_types=4, episode_len=8, p_move=0.5, move_lo=0.25, move_hi=0.75)
TINY_SPLITS = {"model_train": 16, "model_val": 8, "probe_train": 12, "probe_val": 8, "probe_test": 8}


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("grid5")
    cfg = DataConfig(grid=GridConfig(**TINY_GRID), splits=dict(TINY_SPLITS), seed_base=0, out_dir=str(out))
    generate(cfg)
    return cfg
