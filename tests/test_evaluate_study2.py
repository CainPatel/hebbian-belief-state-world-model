import json
import re

import pytest

from hbwm.probes.evaluate import aggregate_study2, write_outputs_study2

# (test_acc, val_acc, rank, saturated, train_acc). Chance is 0.011, so the degeneracy bar is 0.022.
FAMILIES_BDH = {
    "flat_linear_L3": (0.10, 0.10, None, False, 0.45),
    "query_rank_4_L3": (0.28, 0.29, 4, False, 0.51),
    "shared_query_rank_4_L3": (0.20, 0.21, 4, False, 0.40),
    "mlp_state_L3": (0.02, 0.015, None, False, 0.99),  # memorized: degenerate by spec 7
    "mlp_rownorm_L3": (0.19, 0.20, None, False, 0.44),  # BDH-only reduction: context, feeds H7
}
FAMILIES_BASE = {
    "flat_linear": (0.17, 0.17, None, False, 0.30),
    "query_rank_4": (0.18, 0.18, 4, True, 0.33),
    "shared_query_rank_4": (0.16, 0.16, 4, True, 0.29),
    "mlp_state": (0.17, 0.18, None, False, 0.35),
}
L2S = ["0.0001", "0.001", "0.01", "0.1"]
# The real BDH `mlp_state` arm: the parameter count the degeneracy paragraph has to name (spec 7).
N_PARAMS = {"mlp_state": 268_476_928}

# The reviewer's scenario for the rank ruling: BDH's own best arm is r=16, both baselines' own best
# is r=1. Keying on the label would score the baselines at BDH's r=16 (0.10) instead of their own
# best (0.24), and would report H6 supported on a comparison that does not hold.
FAMILIES_BDH_RANKS = {
    "flat_linear_L3": (0.10, 0.10, None, False, 0.45),
    "query_rank_1_L3": (0.12, 0.12, 1, False, 0.30),
    "query_rank_4_L3": (0.20, 0.20, 4, False, 0.35),
    "query_rank_16_L3": (0.28, 0.29, 16, False, 0.51),
}
FAMILIES_BASE_RANKS = {
    "flat_linear": (0.17, 0.17, None, False, 0.30),
    "query_rank_1": (0.24, 0.25, 1, False, 0.33),
    "query_rank_4": (0.15, 0.15, 4, True, 0.29),
    "query_rank_16": (0.10, 0.10, 16, True, 0.28),
}
RANK_TREE = (("bdh_g100", FAMILIES_BDH_RANKS, {"n_heads": 4, "rows": 2048, "cols": 64}),
             ("lstm", FAMILIES_BASE_RANKS, {"n_heads": 1, "rows": 4, "cols": 350}),
             ("rwkv", FAMILIES_BASE_RANKS, {"n_heads": 1, "rows": 20, "cols": 176}))


def _family_of(label):
    """The `family` field Task 7 writes: `spec.name`, so a concrete rank goes back to `_rank_r`."""
    return re.sub(r"_rank_\d+$", "_rank_r", label.split("_L")[0])


def _write(run_dir, families, shape):
    (run_dir / "probes2").mkdir(parents=True, exist_ok=True)
    for label, (test_acc, val_acc, rank, sat, train_acc) in families.items():
        (run_dir / "probes2" / f"{label}.json").write_text(json.dumps({
            "family": _family_of(label), "rank": rank, "level": 3 if "_L" in label else None,
            "test_acc": test_acc, "best_l2": 0.001, "best_restart": 0,
            "val_acc": {f"{a}/0": val_acc for a in L2S},
            "train_acc": {f"{a}/0": train_acc for a in L2S},
            "n_features": shape["rows"] * shape["cols"] * shape["n_heads"], "n_train": 24000,
            "chance": 0.011, "ceiling": 1.0, "ci95": [test_acc - 0.01, test_acc + 0.01],
            "rank_fraction": None if rank is None else min(1.0, rank / min(shape["rows"], shape["cols"])),
            "saturated": sat, "n_params": N_PARAMS.get(_family_of(label), 1), "n_input": 1,
            "input_kind": "flat", "n_restarts": 1,
            "bucket_acc": {}, "n_val": 100, "n_test": 100,
        }))
    (run_dir / "probes2" / "done.json").write_text(json.dumps({"shape": shape, "chance": 0.011}))


def test_aggregate_study2_picks_the_best_matched_family_and_decides_h5_h6(tmp_path):
    for stem, fams, shape in (("bdh_g100", FAMILIES_BDH, {"n_heads": 4, "rows": 2048, "cols": 64}),
                              ("lstm", FAMILIES_BASE, {"n_heads": 1, "rows": 4, "cols": 350}),
                              ("rwkv", FAMILIES_BASE, {"n_heads": 1, "rows": 20, "cols": 176})):
        for seed in (0, 1, 2):
            _write(tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}", fams, shape)
    agg = aggregate_study2(tmp_path, exp="study2")
    assert agg["h5"]["supported"] is True  # 0.28 vs 0.10 on sigma
    assert agg["h6"]["family"] == "query_rank_r"  # the family, not the rank BDH happened to win at
    assert agg["h6"]["supported"] is True
    assert agg["h6"]["saturated_baselines"] == ["lstm", "rwkv"]
    assert agg["h6"]["artifact_warning"] is True
    assert agg["h7"]["attribute_to_capacity"] is False  # 0.19 vs 0.28 is more than 2 points
    assert agg["h7"]["mlp_family"] == "mlp_rownorm"  # H7 is the reduction, not the matched arm


def test_a_degenerate_arm_is_reported_but_excluded_from_h6(tmp_path):
    """Spec 7: mlp_state here memorizes (train 0.99, val 0.015 < 2 * 0.011) at every l2."""
    for stem, fams, shape in (("bdh_g100", FAMILIES_BDH, {"n_heads": 4, "rows": 2048, "cols": 64}),
                              ("lstm", FAMILIES_BASE, {"n_heads": 1, "rows": 4, "cols": 350}),
                              ("rwkv", FAMILIES_BASE, {"n_heads": 1, "rows": 20, "cols": 176})):
        for seed in (0, 1, 2):
            _write(tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}", fams, shape)
    agg = aggregate_study2(tmp_path, exp="study2")
    assert agg["table"]["bdh_g100"]["mlp_state"]["degenerate"] is True
    assert agg["table"]["lstm"]["mlp_state"]["degenerate"] is False
    assert agg["table"]["bdh_g100"]["flat_linear"]["degenerate"] is False
    # still reported, with the evidence a reader needs to audit the call. The reported value is the
    # seed mean of 0.99, which float64 rounds to 0.9899999999999999, hence approx.
    assert agg["table"]["bdh_g100"]["mlp_state"]["train_acc_at_best"] == pytest.approx(0.99)
    # but not eligible to decide H6
    assert "mlp_state" not in agg["h6"]["eligible_families"]
    assert agg["h6"]["degeneracy_exclusions"]["mlp_state"]["bdh"] is True
    assert agg["h6"]["family"] == "query_rank_r"


def test_the_bdh_only_reductions_are_never_an_h6_arm(tmp_path):
    for stem, fams, shape in (("bdh_g100", FAMILIES_BDH, {"n_heads": 4, "rows": 2048, "cols": 64}),
                              ("lstm", FAMILIES_BASE, {"n_heads": 1, "rows": 4, "cols": 350}),
                              ("rwkv", FAMILIES_BASE, {"n_heads": 1, "rows": 20, "cols": 176})):
        for seed in (0, 1, 2):
            _write(tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}", fams, shape)
    agg = aggregate_study2(tmp_path, exp="study2")
    assert agg["h6"]["family"] not in ("mlp_rownorm", "mlp_randproj")
    assert "mlp_state" in agg["table"]["bdh_g100"]
    assert "mlp_rownorm" in agg["table"]["bdh_g100"]
    assert "mlp_rownorm" not in agg["table"]["lstm"]


def test_write_outputs_study2_emits_a_markdown_table_with_rank_fractions(tmp_path):
    for stem, fams, shape in (("bdh_g100", FAMILIES_BDH, {"n_heads": 4, "rows": 2048, "cols": 64}),
                              ("lstm", FAMILIES_BASE, {"n_heads": 1, "rows": 4, "cols": 350}),
                              ("rwkv", FAMILIES_BASE, {"n_heads": 1, "rows": 20, "cols": 176})):
        for seed in (0, 1, 2):
            _write(tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}", fams, shape)
    agg = aggregate_study2(tmp_path, exp="study2")
    write_outputs_study2(agg, tmp_path / "study2" / "results2")
    md = (tmp_path / "study2" / "results2" / "results.md").read_text()
    assert "| model | family | rank | eff. rank frac |" in md
    assert "saturated" in md and "degenerate" in md and "train acc" in md
    assert "Degeneracy criterion (preregistered, spec 7)" in md
    assert "268,476,928" in md  # read from mlp_state's n_params, not hardcoded prose
    assert "{'mlp_state'" not in md  # the exclusions are prose, not a Python dict repr
    assert "`mlp_state` (degenerate on bdh)" in md
    assert (tmp_path / "study2" / "results2" / "h6.json").exists()


def test_h6_scores_each_model_at_its_own_best_rank_not_at_bdhs(tmp_path):
    """Spec 4.2, 5.2 and 7: rank is a per-model hyperparameter chosen on probe_val, not part of
    family identity. BDH's best arm is r=16 at 0.28; each baseline's own best is r=1 at 0.24, and
    its r=16 arm is 0.10. H6 must compare 0.28 against 0.24, which does not clear the 5-point
    margin -- not against 0.10, which would."""
    for stem, fams, shape in RANK_TREE:
        for seed in (0, 1, 2):
            _write(tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}", fams, shape)
    agg = aggregate_study2(tmp_path, exp="study2")
    assert agg["h6"]["family"] == "query_rank_r"
    assert agg["h6"]["comparators"]["lstm"]["mean"] == pytest.approx(0.24)
    assert agg["h6"]["comparators"]["rwkv"]["mean"] == pytest.approx(0.24)
    assert agg["h6"]["supported"] is False  # 0.28 vs 0.24 is 4 points, under the 5-point margin
    assert agg["table"]["bdh_g100"]["query_rank_r"]["ranks"] == [16, 16, 16]
    assert agg["table"]["lstm"]["query_rank_r"]["ranks"] == [1, 1, 1]
    assert agg["h5"]["supported"] is True  # H5 is within BDH, so it is untouched: 0.28 vs 0.10


def test_a_rank_the_seeds_disagree_on_is_reported_per_seed(tmp_path):
    for stem, fams, shape in RANK_TREE:
        for seed in (0, 1, 2):
            f = dict(fams)
            if stem == "lstm" and seed == 2:  # this seed's own best is r=4, not r=1
                f["query_rank_4"] = (0.15, 0.30, 4, True, 0.29)
            _write(tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}", f, shape)
    agg = aggregate_study2(tmp_path, exp="study2")
    arm = agg["table"]["lstm"]["query_rank_r"]
    assert arm["ranks"] == [1, 1, 4]
    assert arm["rank"] is None  # no single rank speaks for the arm
    assert arm["saturated"] is True  # any seed, so H6's artifact warning cannot be missed
    write_outputs_study2(agg, tmp_path / "study2" / "results2")
    md = (tmp_path / "study2" / "results2" / "results.md").read_text()
    assert "| lstm | query_rank_r | [1, 1, 4] |" in md


def test_exploratory_sigma_structure_files_are_not_probe_arms(tmp_path):
    """Task 12 writes `sigma_structure_L<level>.json` into the same probes2/ this globs."""
    for stem, fams, shape in (("bdh_g100", FAMILIES_BDH, {"n_heads": 4, "rows": 2048, "cols": 64}),
                              ("lstm", FAMILIES_BASE, {"n_heads": 1, "rows": 4, "cols": 350}),
                              ("rwkv", FAMILIES_BASE, {"n_heads": 1, "rows": 20, "cols": 176})):
        for seed in (0, 1, 2):
            run_dir = tmp_path / "study2" / f"{stem}_lr0.003" / f"seed{seed}"
            _write(run_dir, fams, shape)
            if stem == "bdh_g100":
                (run_dir / "probes2" / "sigma_structure_L3.json").write_text(json.dumps(
                    {"level": 3, "participation_ratio": 12.5, "decides_nothing": True}))
    agg = aggregate_study2(tmp_path, exp="study2")
    assert "sigma_structure" not in agg["table"]["bdh_g100"]
    assert agg["h6"]["family"] == "query_rank_r"
    # dropped from the arm table, but still carried in the exploratory section
    assert "bdh_g100/seed0/sigma_structure_L3" in agg["structure"]


def test_an_empty_tree_names_the_directory_it_read(tmp_path):
    with pytest.raises(RuntimeError, match="no probe family present on all three seeds") as e:
        aggregate_study2(tmp_path, exp="study2")
    assert "probes2" in str(e.value) and "--exp" in str(e.value)
