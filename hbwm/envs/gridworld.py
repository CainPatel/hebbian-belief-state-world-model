import dataclasses

import numpy as np

EMPTY, WALL = 0, 1
ACTION_DELTAS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int64)  # N, E, S, W as (dx, dy)
ACTION_NAMES = ["N", "E", "S", "W"]


def obj_code(k: int) -> int:
    return 2 + int(k)


def _default_mix():
    return {"random_walk": 1 / 3, "sweep": 1 / 3, "waypoint": 1 / 3}


@dataclasses.dataclass
class GridConfig:
    size: int = 9
    n_objects: int = 3
    n_object_types: int = 4
    episode_len: int = 96
    p_move: float = 0.5
    move_lo: float = 0.25
    move_hi: float = 0.75
    policy_mix: dict = dataclasses.field(default_factory=_default_mix)


class GridWorld:
    """G x G open room with outer walls, typed static landmark objects, one agent. y grows downward."""

    def __init__(self, cfg: GridConfig, seed: int):
        assert cfg.n_objects <= cfg.n_object_types, "objects must have distinct types"
        assert cfg.n_objects + 1 <= cfg.size * cfg.size
        self.cfg = cfg
        self.G = cfg.size
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        G, n = self.G, self.cfg.n_objects
        cells = self.rng.choice(G * G, size=n + 1, replace=False)
        self.agent = np.array([cells[0] % G, cells[0] // G], dtype=np.int64)
        self.obj_pos = np.stack([cells[1:] % G, cells[1:] // G], axis=1).astype(np.int64)
        self.obj_type = self.rng.choice(self.cfg.n_object_types, size=n, replace=False).astype(np.int64)
        self.t = 0

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.G and 0 <= y < self.G

    def step(self, action: int) -> None:
        nxt = self.agent + ACTION_DELTAS[action]
        if self.in_bounds(int(nxt[0]), int(nxt[1])):
            self.agent = nxt
        self.t += 1

    def cell_code(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return WALL
        for k in range(self.cfg.n_objects):
            if self.obj_pos[k, 0] == x and self.obj_pos[k, 1] == y:
                return obj_code(self.obj_type[k])
        return EMPTY

    def window(self) -> np.ndarray:
        ax, ay = int(self.agent[0]), int(self.agent[1])
        return np.array(
            [self.cell_code(ax + dx, ay + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int64
        )

    def visible(self) -> np.ndarray:
        return np.abs(self.obj_pos - self.agent[None, :]).max(axis=1) <= 1

    def cell_in_view(self, x: int, y: int) -> bool:
        return max(abs(x - int(self.agent[0])), abs(y - int(self.agent[1]))) <= 1

    def occupied(self, x: int, y: int) -> bool:
        if self.agent[0] == x and self.agent[1] == y:
            return True
        return bool(((self.obj_pos[:, 0] == x) & (self.obj_pos[:, 1] == y)).any())

    def move_candidates(self) -> list[int]:
        vis = self.visible()
        return [k for k in range(self.cfg.n_objects) if not vis[k]]

    def free_out_of_view_cells(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.G)
            for x in range(self.G)
            if not self.cell_in_view(x, y) and not self.occupied(x, y)
        ]

    def move_object(self, k: int, xy) -> None:
        self.obj_pos[k] = np.array(xy, dtype=np.int64)
