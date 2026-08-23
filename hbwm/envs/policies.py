import numpy as np

from hbwm.envs.gridworld import GridWorld

N, E, S, W = 0, 1, 2, 3
POLICY_NAMES = ["random_walk", "sweep", "waypoint"]


class RandomWalk:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, world: GridWorld) -> int:
        return int(self.rng.integers(4))


class Waypoint:
    """Pick a uniform random target cell != current; walk a shortest path (random axis order)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.target = None

    def act(self, world: GridWorld) -> int:
        if self.target is None or (self.target == world.agent).all():
            while True:
                cand = np.array([self.rng.integers(world.G), self.rng.integers(world.G)], dtype=np.int64)
                if not (cand == world.agent).all():
                    self.target = cand
                    break
        dx = int(self.target[0] - world.agent[0])
        dy = int(self.target[1] - world.agent[1])
        options = []
        if dx > 0:
            options.append(E)
        if dx < 0:
            options.append(W)
        if dy > 0:
            options.append(S)
        if dy < 0:
            options.append(N)
        return int(self.rng.choice(options))


class Sweep:
    """Shortest path to (0,0), then a boustrophedon over all rows, then random walk.
    Worst case on 9x9: 16 + 80 = 96 moves to cover every cell."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.plan = None
        self.i = 0

    def _build(self, world: GridWorld) -> list[int]:
        G = world.G
        plan = [W] * int(world.agent[0]) + [N] * int(world.agent[1])
        for y in range(G):
            plan += [E if y % 2 == 0 else W] * (G - 1)
            if y < G - 1:
                plan.append(S)
        return plan

    def act(self, world: GridWorld) -> int:
        if self.plan is None:
            self.plan = self._build(world)
        if self.i < len(self.plan):
            a = self.plan[self.i]
            self.i += 1
            return a
        return int(self.rng.integers(4))


def make_policy(name: str, rng: np.random.Generator):
    if name == "random_walk":
        return RandomWalk(rng)
    if name == "sweep":
        return Sweep(rng)
    if name == "waypoint":
        return Waypoint(rng)
    raise ValueError(f"unknown policy {name}")
