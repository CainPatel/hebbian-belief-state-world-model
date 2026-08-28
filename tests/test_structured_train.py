import numpy as np
import torch

from hbwm.probes.probe import accuracy, feature_stats, predict_proba
from hbwm.probes.structured import (
    FamilySpec,
    StateShape,
    derot_feature_stats,
    evaluate_on,
    train_family_probes,
)

SHAPE = StateShape(n_heads=1, rows=4, cols=3, rotary=False)  # 12 features
C = 4


def oracle_dataset(n=240, seed=0):
    """One-hot features that name the class: any working probe must reach 100%."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, size=n)
    X = np.zeros((n, SHAPE.n_features), dtype=np.float32)
    X[np.arange(n), y] = 1.0
    return X, y


def test_oracle_features_reach_100_percent():
    X, y = oracle_dataset()
    probes = train_family_probes(X, y, C, SHAPE, FamilySpec("flat_linear"), [1e-4], epochs=40, seed=0,
                                 lr=1e-2, batch=64)
    p = probes[(1e-4, 0)]
    assert accuracy(predict_proba(p, X), y) == 1.0


def test_shuffled_labels_stay_near_chance_on_held_out_data():
    X, y = oracle_dataset(n=400, seed=1)
    rng = np.random.default_rng(2)
    y_shuf = rng.permutation(y)
    probes = train_family_probes(X[:300], y_shuf[:300], C, SHAPE, FamilySpec("flat_linear"), [1e-2],
                                 epochs=20, seed=0, lr=1e-2, batch=64)
    acc = accuracy(predict_proba(probes[(1e-2, 0)], X[300:]), y_shuf[300:])
    assert acc < 0.5  # chance is 0.25; a linear probe on 12 one-hot features cannot memorize noise


def test_restarts_produce_distinct_initializations_and_are_all_returned():
    X, y = oracle_dataset(n=120)
    spec = FamilySpec("query_rank_r", 2, n_restarts=3)
    probes = train_family_probes(X, y, C, SHAPE, spec, [1e-4, 1e-3], epochs=2, seed=0)
    assert set(probes) == {(l2, r) for l2 in (1e-4, 1e-3) for r in range(3)}
    a, b = probes[(1e-4, 0)].q.detach(), probes[(1e-4, 1)].q.detach()
    assert not torch.allclose(a, b, atol=1e-6)


def test_rank_r_probe_learns_the_oracle_too():
    X, y = oracle_dataset(n=240)
    spec = FamilySpec("query_rank_r", SHAPE.saturation_rank, n_restarts=1)
    probes = train_family_probes(X, y, C, SHAPE, spec, [1e-4], epochs=100, seed=0, lr=1e-2, batch=64)
    # >= 0.99 rather than == 1.0: the factorized objective is nonconvex, so this asserts that a
    # saturated-rank probe learns the oracle, not that it converges to the same point every time.
    assert accuracy(predict_proba(probes[(1e-4, 0)], X), y) >= 0.99


def test_positions_are_required_only_by_derot_families():
    X, y = oracle_dataset(n=120)
    freqs = torch.rand(SHAPE.rows)
    pos = np.arange(len(y), dtype=np.float32)
    probes = train_family_probes(X, y, C, SHAPE._replace_rotary(True), FamilySpec("derot_flat_linear"),
                                 [1e-3], positions=pos, epochs=2, seed=0, freqs=freqs)
    assert (1e-3, 0) in probes


def test_derot_family_fits_its_statistics_in_the_derotated_frame():
    """Spec 4.4: derotated families get their own mean/std, and the inner probe is the identity."""
    X, y = oracle_dataset(n=120)
    rot = SHAPE._replace_rotary(True)
    freqs = torch.rand(rot.rows)
    pos = np.arange(len(y), dtype=np.float32)
    p = train_family_probes(X, y, C, rot, FamilySpec("derot_flat_linear"), [1e-3], positions=pos,
                            epochs=1, seed=0, freqs=freqs)[(1e-3, 0)]
    m, s = derot_feature_stats(X, pos, rot, freqs)
    assert torch.allclose(p.mean, torch.from_numpy(m), atol=1e-5)
    assert torch.allclose(p.std, torch.from_numpy(s), atol=1e-5)
    plain_mean, _ = feature_stats(X)
    assert not np.allclose(m, plain_mean, atol=1e-4)  # genuinely a different frame
    assert torch.allclose(p.inner.mean, torch.zeros_like(p.inner.mean))
    assert torch.allclose(p.inner.std, torch.ones_like(p.inner.std))


def test_evaluate_on_scores_every_probe_in_one_pass():
    X, y = oracle_dataset(n=240)
    probes = train_family_probes(X, y, C, SHAPE, FamilySpec("flat_linear"), [1e-4, 1e-1], epochs=40,
                                 seed=0, lr=1e-2, batch=64)
    accs = evaluate_on(probes, X, y)
    assert set(accs) == set(probes)
    assert accs[(1e-4, 0)] == 1.0  # matches the oracle test above, computed in one chunked pass
    assert all(0.0 <= v <= 1.0 for v in accs.values())


def test_derot_family_without_positions_is_an_error():
    X, y = oracle_dataset(n=40)
    import pytest

    with pytest.raises(ValueError):
        train_family_probes(X, y, C, SHAPE._replace_rotary(True), FamilySpec("derot_flat_linear"),
                            [1e-3], epochs=1, seed=0, freqs=torch.rand(SHAPE.rows))
