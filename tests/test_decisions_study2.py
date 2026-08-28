import numpy as np

from hbwm.probes.decisions import (
    h5_decision,
    h6_decision,
    h7_attribution,
    h8_latency,
    is_degenerate,
)


def test_h5_needs_both_the_margin_and_every_paired_difference():
    assert h5_decision([0.20, 0.21, 0.22], [0.10, 0.11, 0.12])["supported"] is True
    # margin met on the mean, but one seed goes the wrong way
    assert h5_decision([0.20, 0.21, 0.05], [0.10, 0.11, 0.12])["supported"] is False
    # every paired difference positive, but the mean margin is only 2 points
    assert h5_decision([0.12, 0.13, 0.14], [0.10, 0.11, 0.12])["supported"] is False


def test_h6_requires_both_baselines_and_reports_the_kill_criterion():
    d = h6_decision([0.30, 0.31, 0.32], {"lstm": [0.17, 0.17, 0.18], "rwkv": [0.21, 0.22, 0.22]},
                    family="query_rank_4", saturated={"lstm": False, "rwkv": False})
    assert d["supported"] is True and d["kill_criterion_fired"] is False
    assert d["family"] == "query_rank_4"

    d = h6_decision([0.10, 0.11, 0.10], {"lstm": [0.17, 0.17, 0.18], "rwkv": [0.21, 0.22, 0.22]},
                    family="flat_linear", saturated={"lstm": False, "rwkv": False})
    assert d["supported"] is False and d["kill_criterion_fired"] is True


def test_h6_flags_a_win_over_a_saturated_baseline_arm():
    d = h6_decision([0.30, 0.31, 0.32], {"lstm": [0.17, 0.17, 0.18], "rwkv": [0.21, 0.22, 0.22]},
                    family="query_rank_16", saturated={"lstm": True, "rwkv": False})
    assert d["supported"] is True
    assert d["saturated_baselines"] == ["lstm"]
    assert d["artifact_warning"] is True


def test_h7_attributes_to_capacity_when_the_mlp_is_within_two_points():
    assert h7_attribution([0.30, 0.30, 0.30], [0.31, 0.31, 0.31])["attribute_to_capacity"] is True
    assert h7_attribution([0.20, 0.20, 0.20], [0.31, 0.31, 0.31])["attribute_to_capacity"] is False


def _episode(ep, steps, visible, flip_at):
    """Build one episode's H8 rows: p_new > p_old from `flip_at` onward (None = never)."""
    n = len(steps)
    p_old = np.full(n, 0.5)
    p_new = np.full(n, 0.1)
    if flip_at is not None:
        p_new[np.asarray(steps) >= flip_at] = 0.9
    return dict(ep=np.full(n, ep), steps=np.asarray(steps), visible=np.asarray(visible),
                p_old=p_old, p_new=p_new)


def _stack(episodes):
    keys = ("ep", "steps", "visible", "p_old", "p_new")
    return {k: np.concatenate([e[k] for e in episodes]) for k in keys}


def test_h8_rebaselines_the_clock_to_the_first_not_visible_step():
    # visible at steps 0 and 1, so t0 = 2; the flip at step 5 is a latency of 3, which passes.
    d = _stack([_episode(0, [0, 1, 2, 3, 4, 5], [True, True, False, False, False, False], flip_at=5)])
    r = h8_latency(d["p_old"], d["p_new"], d["steps"], d["ep"], d["visible"])
    assert r["latencies"] == [3] and r["frac_le5"] == 1.0 and r["supported"] is True


def test_h8_excludes_episodes_that_never_leave_the_window():
    always_visible = _episode(0, [0, 1, 2], [True, True, True], flip_at=None)
    normal = _episode(1, [0, 1, 2], [False, False, False], flip_at=0)
    d = _stack([always_visible, normal])
    r = h8_latency(d["p_old"], d["p_new"], d["steps"], d["ep"], d["visible"])
    assert r["n_excluded"] == 1 and r["n_episodes"] == 1
    assert r["frac_le5"] == 1.0  # the excluded episode is NOT a failure
    assert r["excluded_frac"] == 0.5 and r["low_coverage"] is True


def test_h8_counts_a_never_flipping_episode_with_a_t0_as_a_failure():
    d = _stack([_episode(0, [0, 1, 2], [False, False, False], flip_at=None),
                _episode(1, [0, 1, 2], [False, False, False], flip_at=0)])
    r = h8_latency(d["p_old"], d["p_new"], d["steps"], d["ep"], d["visible"])
    assert r["n_excluded"] == 0 and r["n_episodes"] == 2
    assert r["frac_le5"] == 0.5 and r["supported"] is False


def test_degenerate_arm_is_flagged():
    """Train above 0.95 and val below twice chance at EVERY l2: a memorized arm (spec 7)."""
    train = {"0.0001/0": 0.99, "0.001/0": 0.98, "0.01/0": 0.97, "0.1/0": 0.96}
    val = {"0.0001/0": 0.015, "0.001/0": 0.018, "0.01/0": 0.012, "0.1/0": 0.019}
    d = is_degenerate(train, val, chance=0.011)
    assert d["degenerate"] is True
    assert d["val_bar"] == 0.022
    assert all(v["memorizing"] for v in d["per_l2"].values())


def test_healthy_arm_is_not_flagged():
    train = {"0.0001/0": 0.42, "0.001/0": 0.38, "0.01/0": 0.31, "0.1/0": 0.22}
    val = {"0.0001/0": 0.18, "0.001/0": 0.19, "0.01/0": 0.17, "0.1/0": 0.14}
    assert is_degenerate(train, val, chance=0.011)["degenerate"] is False


def test_one_usable_l2_rescues_an_otherwise_memorizing_arm():
    """The criterion needs the condition at EVERY l2, so a single healthy setting is enough."""
    train = {"0.0001/0": 0.99, "0.001/0": 0.98, "0.01/0": 0.97, "0.1/0": 0.55}
    val = {"0.0001/0": 0.015, "0.001/0": 0.018, "0.01/0": 0.012, "0.1/0": 0.16}
    d = is_degenerate(train, val, chance=0.011)
    assert d["degenerate"] is False
    assert d["per_l2"]["0.1"]["memorizing"] is False


def test_restart_selection_uses_the_best_validation_restart_at_each_l2():
    train = {"0.001/0": 0.99, "0.001/1": 0.60, "0.01/0": 0.99, "0.01/1": 0.99}
    val = {"0.001/0": 0.010, "0.001/1": 0.20, "0.01/0": 0.010, "0.01/1": 0.012}
    d = is_degenerate(train, val, chance=0.011)
    assert d["per_l2"]["0.001"]["key"] == "0.001/1"  # the restart selection would pick
    assert d["degenerate"] is False


def test_degeneracy_on_an_empty_arm_is_false():
    assert is_degenerate({}, {}, chance=0.011)["degenerate"] is False


def test_h8_low_coverage_flag_trips_only_above_25_percent():
    eps = [_episode(i, [0, 1], [False, False], flip_at=0) for i in range(9)]
    eps.append(_episode(9, [0, 1], [True, True], flip_at=None))
    d = _stack(eps)
    r = h8_latency(d["p_old"], d["p_new"], d["steps"], d["ep"], d["visible"])
    assert r["excluded_frac"] == 0.1 and r["low_coverage"] is False
