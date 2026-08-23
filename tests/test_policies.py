import numpy as np

from hbwm.envs.gridworld import GridConfig, GridWorld
from hbwm.envs.policies import POLICY_NAMES, Sweep, Waypoint, make_policy


def test_all_policies_emit_valid_actions():
    for name in POLICY_NAMES:
        w = GridWorld(GridConfig(), 1)
        pol = make_policy(name, np.random.default_rng(1))
        for _ in range(50):
            a = pol.act(w)
            assert a in (0, 1, 2, 3)
            w.step(a)


def test_sweep_covers_every_cell_within_96_steps_on_9x9():
    for seed in range(20):
        w = GridWorld(GridConfig(size=9, episode_len=96), seed)
        pol = Sweep(np.random.default_rng(seed))
        visited = {tuple(w.agent)}
        for _ in range(96):
            w.step(pol.act(w))
            visited.add(tuple(w.agent))
        assert len(visited) == 81, f"seed {seed} visited {len(visited)}"


def test_waypoint_moves_toward_target():
    w = GridWorld(GridConfig(), 2)
    pol = Waypoint(np.random.default_rng(2))
    reached = 0
    for _ in range(300):
        before = w.agent.copy()
        a = pol.act(w)
        tgt = pol.target.copy()
        d0 = np.abs(tgt - before).sum()
        w.step(a)
        d1 = np.abs(tgt - w.agent).sum()
        assert d1 == d0 - 1
        if (w.agent == tgt).all():
            reached += 1
    assert reached >= 5
