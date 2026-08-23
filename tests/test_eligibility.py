import numpy as np

from hbwm.envs.dataset import EpisodeData
from hbwm.probes.eligibility import (BUCKET_NAMES, PairSet, bucket_of, cell_id, eligible_mask, h3_pairs,
                                     last_seen_cell, sample_pairs)


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


def test_h3_pairs(tiny_data):
    d = EpisodeData(tiny_data.out_dir, "probe_train")
    h = h3_pairs(d)
    eps = np.where(d.moved & (d.reobserved_t >= 0))[0]
    assert len(h) == sum(d.L + 1 - d.reobserved_t[e] for e in eps)
    if len(h):
        assert (h.t >= d.reobserved_t[h.ep]).all() and (h.obj == d.move_obj[h.ep]).all()
        assert (h.old_cell == cell_id(d.move_from[h.ep], d.G)).all()
        assert (h.steps_since_reobs == h.t - d.reobserved_t[h.ep]).all()
