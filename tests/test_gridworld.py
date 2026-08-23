import numpy as np

from hbwm.envs.gridworld import EMPTY, WALL, GridConfig, GridWorld, obj_code


def test_seed_determinism():
    a, b = GridWorld(GridConfig(), 7), GridWorld(GridConfig(), 7)
    assert (a.agent == b.agent).all() and (a.obj_pos == b.obj_pos).all() and (a.obj_type == b.obj_type).all()
    c = GridWorld(GridConfig(), 8)
    assert not ((a.agent == c.agent).all() and (a.obj_pos == c.obj_pos).all())


def test_objects_distinct_cells_and_types():
    w = GridWorld(GridConfig(n_objects=4, n_object_types=4), 3)
    cells = {tuple(p) for p in w.obj_pos} | {tuple(w.agent)}
    assert len(cells) == 5
    assert len(set(w.obj_type.tolist())) == 4


def test_window_at_corner_shows_walls():
    w = GridWorld(GridConfig(), 0)
    w.agent = np.array([0, 0])
    w.obj_pos[:] = np.array([[5, 5], [6, 6], [7, 7]])
    win = w.window()
    assert win.tolist() == [WALL, WALL, WALL, WALL, EMPTY, EMPTY, WALL, EMPTY, EMPTY]


def test_window_shows_object_at_right_index():
    w = GridWorld(GridConfig(), 0)
    w.agent = np.array([4, 4])
    w.obj_pos[:] = np.array([[5, 4], [0, 0], [8, 8]])  # first object east of agent -> index 5
    w.obj_type[:] = np.array([2, 0, 1])
    win = w.window()
    assert win[5] == obj_code(2) and win[4] == EMPTY
    assert w.visible().tolist() == [True, False, False]


def test_bump_is_noop_and_t_increments():
    w = GridWorld(GridConfig(), 0)
    w.agent = np.array([0, 0])
    w.step(3)  # W
    w.step(0)  # N
    assert w.agent.tolist() == [0, 0] and w.t == 2
    w.step(1)  # E
    w.step(2)  # S
    assert w.agent.tolist() == [1, 1]


def test_free_out_of_view_cells_excludes_view_and_occupied():
    w = GridWorld(GridConfig(size=5, n_objects=1), 0)
    w.agent = np.array([2, 2])
    w.obj_pos[:] = np.array([[0, 0]])
    free = w.free_out_of_view_cells()
    assert (0, 0) not in free and (2, 2) not in free and (3, 3) not in free and (0, 4) in free
    assert len(free) == 25 - 9 - 1
    assert w.move_candidates() == [0]
    w.move_object(0, (4, 4))
    assert w.obj_pos[0].tolist() == [4, 4]
