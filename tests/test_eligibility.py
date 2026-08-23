from types import SimpleNamespace

import numpy as np

from hbwm.envs.dataset import EpisodeData
from hbwm.envs.episode import steps_since_seen as _steps_since_seen
from hbwm.probes.eligibility import (
    BUCKET_NAMES,
    PairSet,
    bucket_of,
    cell_id,
    eligible_mask,
    h3_pairs,
    last_seen_cell,
    sample_pairs,
)


def _mk_synthetic(n, n_obj, L, G, visible_spec):
    """Minimal EpisodeData-like stand-in with just the attributes sample_pairs/last_seen_cell need.

    visible_spec[k] is "once" (object k visible only at t=0, so steps_since_seen == t thereafter,
    sweeping every bucket over a long horizon) or "always" (object k never eligible).
    """
    T = L + 1
    visible = np.zeros((n, T, n_obj), dtype=bool)
    obj_pos = np.zeros((n, T, n_obj, 2), dtype=np.int64)
    for k, spec in enumerate(visible_spec):
        if spec == "once":
            visible[:, 0, k] = True
        elif spec == "always":
            visible[:, :, k] = True
        else:
            raise ValueError(spec)
        for e in range(n):
            obj_pos[e, :, k] = [(e + k + 1) % G, (e + k + 2) % G]
    sss = np.stack([_steps_since_seen(visible[e]) for e in range(n)])
    stale = np.zeros((n, T, n_obj), dtype=bool)
    return SimpleNamespace(n=n, L=L, G=G, n_obj=n_obj, visible=visible, steps_since_seen=sss, stale=stale,
                            obj_pos=obj_pos)


def test_bucket_of():
    assert bucket_of(np.array([-1, 0, 1, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 200])).tolist() == \
        [-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert len(BUCKET_NAMES) == 6


def test_eligibility_and_oracle(tiny_data):
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    el = eligible_mask(d)
    assert el.shape == d.visible.shape
    assert not (el & d.visible).any() and not (el & d.stale).any() and (d.steps_since_seen[el] >= 1).all()
    oracle = last_seen_cell(d)
    for ep in range(d.n):
        for k in range(d.n_obj):
            last = -1
            for t in range(d.L + 1):
                if d.visible[ep, t, k]:
                    last = cell_id(d.obj_pos[ep, t, k], d.G)
                assert oracle[ep, t, k] == last


def test_sample_pairs(tiny_data, tmp_path):
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    ps = sample_pairs(d, np.random.default_rng(0), per_obj=3)
    assert len(ps) > 0
    el = eligible_mask(d)
    assert el[ps.ep, ps.t, ps.obj].all()
    assert (ps.label == cell_id(d.obj_pos[ps.ep, ps.t, ps.obj], d.G)).all()
    assert (ps.bucket == bucket_of(ps.sss)).all() and (ps.bucket >= 0).all()
    for ep in np.unique(ps.ep):
        for k in range(d.n_obj):
            assert ((ps.ep == ep) & (ps.obj == k)).sum() <= 3
    ps.save(tmp_path / "p.npz")
    back = PairSet.load(tmp_path / "p.npz")
    assert (back.ep == ps.ep).all() and len(back.subset(np.arange(2))) == 2


def test_sample_pairs_round_robin_over_all_buckets():
    # obj 0 is visible only at t=0 over a long horizon, so steps_since_seen == t for t >= 1 and every
    # bucket (1-4, 5-8, 9-16, 17-32, 33-64, 65+) is populated; obj 1 is always visible, so never eligible.
    L, G = 100, 5
    d = _mk_synthetic(n=1, n_obj=2, L=L, G=G, visible_spec=["once", "always"])

    # (a) round-robin balance: per_obj=12 splits evenly (2 per bucket); per_obj=8 leaves the
    # deterministic remainder in the first two buckets (round-robin visits buckets 0..5 in order).
    ps12 = sample_pairs(d, np.random.default_rng(0), per_obj=12)
    assert len(ps12) == 12 and (ps12.obj == 0).all()
    assert {b: int((ps12.bucket == b).sum()) for b in range(6)} == {b: 2 for b in range(6)}

    ps8 = sample_pairs(d, np.random.default_rng(0), per_obj=8)
    assert len(ps8) == 8
    assert {b: int((ps8.bucket == b).sum()) for b in range(6)} == {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}

    # (b) a fixed rng gives an identical PairSet on replay; a different rng changes which timesteps
    # are drawn from each bucket.
    ps8_replay = sample_pairs(d, np.random.default_rng(0), per_obj=8)
    for f in ("ep", "t", "obj", "label", "bucket", "oracle", "sss"):
        assert (getattr(ps8, f) == getattr(ps8_replay, f)).all()
    ps8_seed1 = sample_pairs(d, np.random.default_rng(1), per_obj=8)
    assert ps8_seed1.t.tolist() != ps8.t.tolist()

    # (c) oracle labels agree with a direct last_seen_cell lookup at the sampled (ep, t, obj).
    assert (ps8.oracle == last_seen_cell(d)[ps8.ep, ps8.t, ps8.obj]).all()

    # (d) an object that is never eligible contributes no pairs.
    assert not (ps12.obj == 1).any() and not (ps8.obj == 1).any()

    # A fully-ineligible dataset yields an empty, correctly-typed PairSet.
    d_none = _mk_synthetic(n=1, n_obj=1, L=10, G=G, visible_spec=["always"])
    ps_none = sample_pairs(d_none, np.random.default_rng(0), per_obj=5)
    assert len(ps_none) == 0
    for f in ("ep", "t", "obj", "label", "bucket", "oracle", "sss"):
        assert getattr(ps_none, f).dtype == np.int64


def test_h3_pairs(tiny_data):
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    h = h3_pairs(d)
    eps = np.where(d.moved & (d.reobserved_t >= 0))[0]
    assert len(h) == sum(d.L + 1 - d.reobserved_t[e] for e in eps)
    if len(h):
        assert (h.t >= d.reobserved_t[h.ep]).all() and (h.obj == d.move_obj[h.ep]).all()
        assert (h.old_cell == cell_id(d.move_from[h.ep], d.G)).all()
        assert (h.new_cell == cell_id(d.move_to[h.ep], d.G)).all()
        assert (h.steps_since_reobs == h.t - d.reobserved_t[h.ep]).all()
        assert (h.visible_now == d.visible[h.ep, h.t, h.obj]).all()
