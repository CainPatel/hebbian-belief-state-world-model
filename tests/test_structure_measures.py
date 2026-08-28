import numpy as np
import torch

from hbwm.instrument.structure import (
    effective_rank,
    participation_ratio,
    row_norm_stats,
    write_concentration,
)


def test_participation_ratio_endpoints():
    one_hot = torch.zeros(1, 8)
    one_hot[0, 3] = 5.0
    assert torch.allclose(
        participation_ratio(one_hot), torch.tensor([1.0], dtype=torch.float64), atol=1e-5
    )
    assert torch.allclose(
        participation_ratio(torch.ones(1, 2048)),
        torch.tensor([2048.0], dtype=torch.float64),
        atol=1e-3,
    )


def test_row_norm_stats_on_a_single_loaded_row():
    sigma = torch.zeros(1, 1, 100, 4)
    sigma[0, 0, 7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    r = row_norm_stats(sigma)
    assert r["participation_ratio"]["median"] == 1.0
    assert r["frac_below_1pct_of_max"]["median"] == 0.99
    assert r["frac_below_10pct_of_max"]["median"] == 0.99


def test_effective_rank_of_a_rank_one_matrix_is_one():
    u = torch.randn(1, 1, 32, 1)
    v = torch.randn(1, 1, 1, 4)
    r = effective_rank(u @ v)
    assert r["n_components_90pct"]["median"] == 1.0
    assert r["n_components_99pct"]["median"] == 1.0
    assert abs(r["participation_ratio"]["median"] - 1.0) < 1e-3


def test_effective_rank_of_an_isotropic_matrix_is_full():
    q, _ = torch.linalg.qr(torch.randn(32, 4))
    sigma = q.reshape(1, 1, 32, 4)  # 4 equal singular values
    r = effective_rank(sigma)
    assert r["n_components_99pct"]["median"] == 4.0
    assert abs(r["participation_ratio"]["median"] - 4.0) < 1e-3


def test_write_concentration_shares_are_monotone_and_bounded():
    w = torch.zeros(1, 1, 2048)
    w[0, 0, :20] = 1.0  # exactly the top 1 percent (floor(2048 * 0.01) = 20)
    r = write_concentration(w)
    assert abs(r["share_top_1pct"]["median"] - 1.0) < 1e-6
    assert abs(r["share_top_10pct"]["median"] - 1.0) < 1e-6

    w_flat = torch.ones(1, 1, 2048)
    r = write_concentration(w_flat)
    assert abs(r["share_top_1pct"]["median"] - 20 / 2048) < 1e-6
    assert abs(r["share_top_10pct"]["median"] - 204 / 2048) < 1e-6


def test_all_summaries_carry_median_and_deciles():
    sigma = torch.randn(9, 2, 16, 4)
    for r in (row_norm_stats(sigma), effective_rank(sigma)):
        for v in r.values():
            if not isinstance(v, dict):  # scalar bookkeeping entries such as n_rows
                continue
            assert set(v) >= {"median", "p10", "p90", "min", "max"}
            assert v["p10"] <= v["median"] <= v["p90"] or np.isclose(v["p10"], v["p90"])
