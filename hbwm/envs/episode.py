import dataclasses
import math

import numpy as np

from hbwm.envs.gridworld import GridConfig, GridWorld
from hbwm.envs.policies import POLICY_NAMES, make_policy


@dataclasses.dataclass
class Episode:
    actions: np.ndarray
    agent_pos: np.ndarray
    obj_pos: np.ndarray
    obj_type: np.ndarray
    windows: np.ndarray
    visible: np.ndarray
    moved: bool
    move_t: int
    move_obj: int
    move_from: np.ndarray
    move_to: np.ndarray
    reobserved_t: int
    policy: int
    seed: int


def steps_since_seen(visible: np.ndarray) -> np.ndarray:
    T, n = visible.shape
    out = np.full((T, n), -1, dtype=np.int64)
    last = np.full(n, -1, dtype=np.int64)
    for t in range(T):
        last = np.where(visible[t], t, last)
        out[t] = np.where(last >= 0, t - last, -1)
    return out


def stale_mask(ep) -> np.ndarray:
    """True while a moved object's location is unknowable: t > move_t and not yet re-observed."""
    T, n = ep.visible.shape
    st = np.zeros((T, n), dtype=bool)
    if ep.moved:
        end = ep.reobserved_t if ep.reobserved_t >= 0 else T
        st[ep.move_t + 1 : end, ep.move_obj] = True
    return st


def run_episode(cfg: GridConfig, seed: int) -> Episode:
    ss = np.random.SeedSequence(seed).spawn(3)
    world = GridWorld(cfg, int(ss[0].generate_state(1)[0]))
    rng_p = np.random.default_rng(int(ss[1].generate_state(1)[0]))
    rng_e = np.random.default_rng(int(ss[2].generate_state(1)[0]))

    names = [n for n in POLICY_NAMES if n in cfg.policy_mix]
    probs = np.array([cfg.policy_mix[n] for n in names], dtype=np.float64)
    probs /= probs.sum()
    pidx = int(rng_e.choice(len(names), p=probs))
    policy = make_policy(names[pidx], rng_p)

    L, n = cfg.episode_len, cfg.n_objects
    do_move = bool(rng_e.random() < cfg.p_move)
    lo, hi = math.ceil(cfg.move_lo * L), math.floor(cfg.move_hi * L)
    t_m = int(rng_e.integers(lo, hi + 1)) if do_move else -1

    actions = np.zeros(L, dtype=np.int64)
    agent_pos = np.zeros((L + 1, 2), dtype=np.int64)
    obj_pos = np.zeros((L + 1, n, 2), dtype=np.int64)
    windows = np.zeros((L + 1, 9), dtype=np.int64)
    visible = np.zeros((L + 1, n), dtype=bool)

    def record(t):
        agent_pos[t] = world.agent
        obj_pos[t] = world.obj_pos
        windows[t] = world.window()
        visible[t] = world.visible()

    moved, move_obj = False, -1
    move_from = np.array([-1, -1], dtype=np.int64)
    move_to = np.array([-1, -1], dtype=np.int64)

    record(0)
    for t in range(1, L + 1):
        a = policy.act(world)
        actions[t - 1] = a
        world.step(a)
        record(t)
        if do_move and t == t_m:  # after o_t is emitted, before a_{t+1}
            cands = world.move_candidates()
            free = world.free_out_of_view_cells()
            if cands and free:
                k = cands[int(rng_e.integers(len(cands)))]
                xy = free[int(rng_e.integers(len(free)))]
                move_from = world.obj_pos[k].copy()
                world.move_object(k, xy)
                move_to = world.obj_pos[k].copy()
                moved, move_obj = True, k

    reobserved_t = -1
    if moved:
        later = np.where(visible[t_m + 1 :, move_obj])[0]
        if len(later):
            reobserved_t = int(t_m + 1 + later[0])

    return Episode(
        actions=actions, agent_pos=agent_pos, obj_pos=obj_pos, obj_type=world.obj_type.copy(),
        windows=windows, visible=visible, moved=moved, move_t=t_m if moved else -1, move_obj=move_obj,
        move_from=move_from, move_to=move_to, reobserved_t=reobserved_t, policy=pidx, seed=seed,
    )
