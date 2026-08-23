import numpy as np

from hbwm.envs.episode import run_episode, stale_mask, steps_since_seen
from hbwm.envs.gridworld import GridConfig
from hbwm.envs.policies import POLICY_NAMES


def test_determinism_and_shapes():
    cfg = GridConfig(size=7, n_objects=2, episode_len=20)
    a, b = run_episode(cfg, 5), run_episode(cfg, 5)
    assert (a.actions == b.actions).all() and (a.obj_pos == b.obj_pos).all() and a.moved == b.moved
    assert a.agent_pos.shape == (21, 2) and a.obj_pos.shape == (21, 2, 2) and a.windows.shape == (21, 9)
    assert a.visible.shape == (21, 2) and 0 <= a.policy < len(POLICY_NAMES)


def test_visibility_consistent_with_positions():
    ep = run_episode(GridConfig(episode_len=40), 11)
    cheb = np.abs(ep.obj_pos - ep.agent_pos[:, None, :]).max(axis=2)
    assert (ep.visible == (cheb <= 1)).all()


def test_move_invariants_over_many_seeds():
    cfg = GridConfig(size=9, n_objects=3, episode_len=40, p_move=1.0)
    n_moved = 0
    for seed in range(150):
        ep = run_episode(cfg, seed)
        static = np.ones(3, dtype=bool)
        if ep.moved:
            n_moved += 1
            k, tm = ep.move_obj, ep.move_t
            assert 10 <= tm <= 30
            assert not ep.visible[tm, k]  # out of view before
            assert (ep.obj_pos[tm, k] == ep.move_from).all() and (ep.obj_pos[tm + 1, k] == ep.move_to).all()
            assert np.abs(ep.move_to - ep.agent_pos[tm]).max() > 1  # new cell out of view at tm
            assert (ep.obj_pos[: tm + 1, k] == ep.move_from).all() and (ep.obj_pos[tm + 1 :, k] == ep.move_to).all()
            static[k] = False
            if ep.reobserved_t >= 0:
                assert ep.reobserved_t > tm and ep.visible[ep.reobserved_t, k]
                assert not ep.visible[tm + 1 : ep.reobserved_t, k].any()
        else:
            assert ep.move_obj == -1 and ep.reobserved_t == -1
        for j in np.where(static)[0]:
            assert (ep.obj_pos[:, j] == ep.obj_pos[0, j]).all()
    assert n_moved >= 100


def test_steps_since_seen_and_stale_hand_example():
    vis = np.array([[1, 0], [0, 0], [0, 1], [1, 0]], dtype=bool)
    sss = steps_since_seen(vis)
    assert sss.tolist() == [[0, -1], [1, -1], [2, 0], [0, 1]]

    class EP:  # minimal stand-in
        moved, move_obj, move_t, reobserved_t = True, 1, 1, 3
        visible = np.zeros((6, 2), dtype=bool)

    st = stale_mask(EP)
    assert st[:, 0].sum() == 0
    assert st[:, 1].tolist() == [False, False, True, False, False, False]
    EP.reobserved_t = -1
    assert stale_mask(EP)[:, 1].tolist() == [False, False, True, True, True, True]
