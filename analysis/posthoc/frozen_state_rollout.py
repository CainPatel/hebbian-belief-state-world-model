"""Frozen-sigma rollout: does the belief SURVIVE without further writes? Post-hoc, EXPLORATORY.

NOT PREREGISTERED. This script cannot revise H1-H8, or any other preregistered decision from
Study 1 or Study 2. It decides nothing. It exists so these exploratory numbers are reproducible.

The question
------------
A cheap version of the imagination study SPEC.md originally planned, and the pivot target the
Study 2 preregistration named if H6 failed. If `sigma` really HOLDS a belief, then freezing it part
way through an episode should preserve the predictions that depend on memory while the predictions
that depend on fresh input degrade. If instead the belief is continuously REFRESHED rather than
stored, freezing should hurt the memory-relevant predictions most.

Related evidence this sits next to: `ykv_causal_patch.py` showed that corrupting the associative
read at the deep levels costs memory-relevant cells far more than the matched no-memory control
(L5 selectivity 5.35x), so the read is doing memory work. That is a test of the read. This is a
test of the WRITE: stop updating the thing being read from, and see which class of prediction
falls over.

The intervention
----------------
`HBWMCore.step(tok, state, plasticity=..., plasticity_scale=...)` gates the sigma update:

  "full"    decay by gamma, then write alpha * k (x) x with alpha = 1.0.
  "frozen"  no decay and no write; sigma stays BIT-IDENTICAL to its value on entry.
  "scaled"  decay applied, write scaled by `plasticity_scale` (0.0 stops writes, keeps decay).

The recurrent path is driven through `hbwm/instrument/recorder.py`'s `SigmaRecorder`, whose
callback receives `logits`. To switch plasticity part way through an episode we monkeypatch the
BOUND METHOD `model.step` on the loaded instance and override its `plasticity` argument as a
function of `state.t`, which is the position about to be consumed. `hbwm/` is never edited, and the
class is never touched, only the one live object. This mirrors the monkeypatch discipline of
`ykv_causal_patch.py`, whose token-class labelling and saturation guard are reused here directly.

Conditions, with `cut` a token position and `T_run = T - 1` the number of driven positions:

  full                 plasticity="full" for the whole episode. The CORRECTNESS CHECK: this is the
                       recurrent path computing exactly what the parallel path computes, so its
                       per-class CEs must match the parallel intact numbers. It is also the
                       baseline every delta below is taken against.
  frozen@f             "full" for positions < cut, "frozen" from cut onward.
  scaled0@f            "full" for positions < cut, "scaled" with plasticity_scale = 0.0 from cut.

  frzL45@f             per-LEVEL freeze: levels 4 and 5 stop updating from `cut` onward while
                       levels 0-3 keep decaying and writing normally.
  frzL5@f              the same for level 5 alone.
  frzL0123@f           the mirror-image control: the SHALLOW levels stop updating, the deep ones
                       keep going.

for f in 25%, 50%, 75% of T_run.

WHY THE PER-LEVEL ARMS EXIST
----------------------------
The wholesale `frozen` arms saturate at every cutoff: freezing all of sigma also removes the read
of the LAST FEW tokens, which carry the agent's current coordinates and the current observation
window, so even class-1 empty/wall cells (which need no memory at all) blow up and the class-2 vs
class-3 contrast becomes a ceiling artefact. That is a real bounded finding -- sigma cannot be
ablated as memory separately from current context by freezing it wholesale -- and the arms are kept
on the record for it. But it is not a test of the belief hypothesis.

`ykv_causal_patch.py` localizes the memory-specific effect to the DEEP levels (patching L5's read
alone leaves overall CE at 0.052 against an intact 0.025 and empty/wall cells untouched at -0.0014,
yet costs +2.59 nats on memory-relevant cells, a 5.35x selectivity, while L0-L3 is where the damage
goes global). So freezing only L4/L5 should leave the shallow, current-context computation running
and escape the ceiling. `frzL0123` is the mirror-image control: if the SHALLOW freeze is the one
that saturates while the deep freeze does not, that confirms the diagnosis of why the wholesale
version failed.

HOW THE PER-LEVEL FREEZE IS IMPLEMENTED
---------------------------------------
`HBWMCore.step`'s `plasticity` argument is GLOBAL, so a level subset cannot be expressed through
it. Instead the write is undone: plasticity stays "full" throughout, and after each step from `cut`
onward the frozen levels' slices of `state.sigma` are overwritten with a clone taken at the cut.
`state.sigma` is aliased by the recorder payload, and `step` mutates it in place, so restoring it
between steps is exactly equivalent to the level never having updated: the READ at the next step
sees the cut-time value, and the write that step performed is discarded before anything else
observes it. `hbwm/` is never edited.

This is the save-before-the-call / restore-after-the-call scheme with one clone TOTAL instead of
one clone per step: since the frozen slice is restored to the cut-time value after every step, the
value at entry to every subsequent step is that same cut-time value, so a single snapshot serves.
The driver asserts, on the first `CHECK_STEPS` frozen steps, that (a) the frozen levels DID write
before the restore, so the restore is doing real work, (b) the frozen slice is bit-identical to the
snapshot after the restore, and (c) an unfrozen level DID change over the same steps. It also
re-checks bit-identity at the final position.

GAMMA NOTE. `scaled0` keeps the decay and stops the writes; `frozen` stops both. At
`decay_gamma = 1.0` there IS no decay, so the two conditions are the same computation and produce
identical numbers by construction. The run `bdh_g100_lr0.003` is gamma = 1.0, so `scaled0` is NOT
independent evidence there and is reported as the identity check it is. The script asserts the
identity when gamma == 1.0 and reports the pair as separable only when gamma < 1.0.

What is measured
----------------
Per-token cross-entropy over a `probe_test` SUBSET (the recurrent path is far slower than the
parallel one; 200 episodes yields on the order of 1,200 class-2 tokens, and the actual count is
printed), aggregated into the same four classes (0 non-window, 1 window empty/wall, 3 object
needing NO memory, 2 object RETURNING after an absence to the cell it was last seen at).

Reported over TWO position windows:

  post-cut   masked positions >= cut ONLY. This is the headline: it is the only region where the
             intervention is active, so the all-positions figure merely dilutes it by the untouched
             prefix. Class counts shrink with the cut fraction and are printed.
  all        every masked position, for completeness.

Overall CE is reported per condition so it is visible which arms stayed under the guard, and the
DIAGNOSIS block states plainly whether the shallow freeze saturates while the deep freeze does not.

THE QUANTITY THAT MATTERS is whether freezing hurts class 2 MORE or LESS than class 3:

    excess = d(class2) - d(class3),  ratio = d(class2) / d(class3),  deltas against `full`.

  excess < 0 (ratio < 1)  freezing preserves memory-relevant predictions RELATIVELY BETTER than the
                          matched no-memory control: consistent with sigma holding a belief that
                          survives without further writes.
  excess > 0 (ratio > 1)  freezing hurts memory-relevant predictions MORE: the belief is being
                          continuously refreshed rather than stored.

Both outcomes are interesting and neither is rounded toward. Any condition whose overall CE exceeds
SATURATION_CE (1.0 nat) is flagged and its contrast discarded rather than reported as a null.

Usage
-----
    uv run python analysis/posthoc/frozen_state_rollout.py                 # full run, writes JSON
    uv run python analysis/posthoc/frozen_state_rollout.py --validate      # seed 0, 20 eps, 1 cut
    uv run python analysis/posthoc/frozen_state_rollout.py --cuts 0.5
    uv run python analysis/posthoc/frozen_state_rollout.py --freeze-sets 4,5   # deep arm only

`HBWM_ROOT` overrides the artifact root (must contain `runs/` and `data/`); it defaults to the
Study 1 worktree. Read-only on checkpoints and data; the single JSON output is the only write.
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hbwm.device import release_memory, select_device
from hbwm.envs.dataset import EpisodeData
from hbwm.instrument.recorder import SigmaRecorder
from hbwm.train import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ykv_causal_patch import (  # noqa: E402
    CLASS_NAMES,
    DATA_DIR,
    N_LAYER,
    ROOT,
    RUN,
    SATURATION_CE,
    SEEDS,
    build_classes,
    check_classes,
)

OUT_PATH = ROOT / "runs/study1/results2/posthoc_frozen_rollout.json"

# The recurrent path costs ~0.30 s/episode at chunk 50 on MPS; 200 episodes gives ~1,200 class-2
# tokens, which is the count that limits the class-2 estimate.
N_EPISODES = 200
CHUNK = 50
CUTS = [0.25, 0.50, 0.75]
BASELINE = "full"

# Level subsets frozen by the per-level arms: the deep pair the causal patching implicated, level 5
# alone, and the shallow mirror-image control.
FREEZE_SETS = [(4, 5), (5,), (0, 1, 2, 3)]

# How many steps after the cut to spend on the bit-identity / did-write / unfrozen-changed checks.
# Each check forces a device sync, so it is deliberately not every step.
CHECK_STEPS = 5

# Per-class CE agreement required between the recurrent `full` condition and the parallel path on
# the SAME episodes. These are the same arithmetic in two orders, so only fp32 reassociation
# separates them; anything larger means the recurrent driver is wrong.
AGREE_ABS = 5e-3
AGREE_REL = 0.02


# ---------------------------------------------------------------- conditions


def family_name(mode: str, levels=None) -> str:
    """The condition family, i.e. the name without the cutoff suffix."""
    return mode if levels is None else "frzL" + "".join(str(v) for v in levels)


def cond_name(mode: str, frac, levels=None) -> str:
    return BASELINE if frac is None else f"{family_name(mode, levels)}@{int(round(100 * frac))}"


def build_conditions(cuts, modes, freeze_sets):
    conds = [(BASELINE, None, None, None)]  # (name, mode, frac, freeze_levels)
    for f in cuts:
        for m in modes:
            conds.append((cond_name(m, f), m, f, None))
        for lv in freeze_sets:
            conds.append((cond_name("levelfrozen", f, lv), "levelfrozen", f, tuple(lv)))
    return conds


class StepPhaser:
    """Wraps the bound `step`, overriding `plasticity` as a function of the position `state.t`.

    Positions strictly below `cut` run "full"; from `cut` onward they run `mode`. `cut is None`
    means "full" throughout. Counts both phases so the driver can assert the schedule actually
    fired, which is what makes the condition labels trustworthy.
    """

    def __init__(self, model, cut, mode, scale):
        self.orig_step = model.step
        self.cut = cut
        self.mode = mode
        self.scale = float(scale)
        self.n_full = 0
        self.n_post = 0

    def step(self, tok, state, plasticity="full", plasticity_scale=1.0):
        if self.cut is None or int(state.t) < self.cut:
            self.n_full += 1
            return self.orig_step(tok, state, plasticity="full", plasticity_scale=1.0)
        self.n_post += 1
        return self.orig_step(tok, state, plasticity=self.mode, plasticity_scale=self.scale)


@contextmanager
def phased(model, cut, mode, scale=0.0):
    """Install the phased-plasticity wrapper for the duration. `hbwm/` is never touched."""
    ph = StepPhaser(model, cut, mode, scale)
    model.step = ph.step  # instance attribute shadows the class method
    try:
        yield ph
    finally:
        del model.step
    assert model.step.__func__ is ph.orig_step.__func__, "failed to restore step"


# ---------------------------------------------------------------- measurement


def _slices_equal(sigma, levels, snap) -> bool:
    """Bit-identity of `sigma`'s frozen level slices against the cut-time snapshot, level by level
    so the comparison never materialises the whole stack."""
    return all(bool(torch.equal(sigma[lv], snap[i])) for i, lv in enumerate(levels))


def _and(acc, v):
    return v if acc is None else (acc and v)


@torch.no_grad()
def run_condition(model, d, indices, cut, mode, scale, freeze_levels, chunk, device):
    """Drives SigmaRecorder over the subset; returns (per-token CE [n_sub, T_run], diagnostics).

    `freeze_levels is None` gives the wholesale arms, gated through `step`'s `plasticity`.
    Otherwise plasticity stays "full" and those levels' sigma slices are restored to their cut-time
    value after every step from `cut` onward, which is exactly "this level stopped updating".
    """
    frz = list(freeze_levels) if freeze_levels else []
    unf = [lv for lv in range(model.hcfg.n_layer) if lv not in frz]
    ce_rows = []
    dg = {
        "n_full": 0,
        "n_post": 0,
        "sigma_ok": None,
        "n_restores": 0,
        "frozen_wrote": None,
        "frozen_identical_after_restore": None,
        "unfrozen_changed": None,
        "frozen_identical_at_end": None,
    }
    for start in range(0, len(indices), chunk):
        idx = indices[start : start + chunk]
        x, y, _ = d.batch_at(idx, device)
        T_run = x.size(1)
        rec = SigmaRecorder(model)
        per_pos = []
        st = {}

        def fn(pos, payload, _y=y, _per_pos=per_pos, _st=st, _T=T_run):
            _per_pos.append(
                F.cross_entropy(payload["logits"].float(), _y[:, pos], reduction="none")
            )
            if cut is None:
                return
            sig = payload["sigma"]
            if not frz:
                # Wholesale arms: spot-check that sigma really is held fixed (level 0, episode 0).
                if pos == cut - 1:
                    _st["w"] = sig[0, 0].clone()
                elif pos == _T - 1 and "w" in _st:
                    _st["w_ok"] = bool(torch.equal(_st["w"], sig[0, 0]))
                return
            if pos == cut - 1:
                # sigma just after the last plastic step == sigma on entry to the frozen phase.
                _st["snap"] = [sig[lv].clone() for lv in frz]
                _st["usnap"] = sig[unf[0]].clone() if unf else None
            elif pos >= cut and "snap" in _st:
                checking = pos < cut + CHECK_STEPS
                if checking:
                    # BEFORE the restore the frozen levels must have written, or the restore is
                    # a no-op and the condition would be indistinguishable from the baseline.
                    _st["wrote"] = _and(
                        _st.get("wrote"), not _slices_equal(sig, frz, _st["snap"])
                    )
                    if _st["usnap"] is not None:
                        _st["uch"] = _and(
                            _st.get("uch"), not bool(torch.equal(sig[unf[0]], _st["usnap"]))
                        )
                for i, lv in enumerate(frz):
                    sig[lv].copy_(_st["snap"][i])
                _st["restores"] = _st.get("restores", 0) + 1
                if checking:
                    _st["ident"] = _and(_st.get("ident"), _slices_equal(sig, frz, _st["snap"]))
                if pos == cut + CHECK_STEPS - 1:
                    _st["usnap"] = None
                if pos == _T - 1:
                    _st["end_ok"] = _slices_equal(sig, frz, _st["snap"])

        ph_cut = None if frz else cut
        with phased(model, ph_cut, mode, scale) as ph:
            rec.run(x, positions=None, fn=fn, plasticity="full")
            dg["n_full"] += ph.n_full
            dg["n_post"] += ph.n_post
        ce_rows.append(torch.stack(per_pos, dim=1).float().cpu().numpy())
        dg["sigma_ok"] = _and(dg["sigma_ok"], st["w_ok"]) if "w_ok" in st else dg["sigma_ok"]
        dg["n_restores"] += st.get("restores", 0)
        for key, src in (
            ("frozen_wrote", "wrote"),
            ("frozen_identical_after_restore", "ident"),
            ("unfrozen_changed", "uch"),
            ("frozen_identical_at_end", "end_ok"),
        ):
            if src in st:
                dg[key] = _and(dg[key], st[src])
        del st, per_pos
        release_memory(device)
    return np.concatenate(ce_rows, axis=0), dg


@torch.no_grad()
def run_parallel(model, d, indices, chunk, device):
    """The parallel `model(tokens[:, :-1])` path on the same episodes, for the correctness check."""
    rows = []
    for start in range(0, len(indices), chunk):
        idx = indices[start : start + chunk]
        x, y, _ = d.batch_at(idx, device)
        logits, _ = model(x)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view_as(y)
        rows.append(ce.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def class_stats(ce, cls_sub, mask_t, lo=0):
    """Per-class mean CE over masked positions >= lo."""
    window = mask_t.copy()
    window[:lo] = False
    sums, counts = {}, {}
    for c in CLASS_NAMES:
        m = (cls_sub == c) & window[None, :]
        counts[c] = int(m.sum())
        sums[c] = float(ce[m].sum())
    total_n = sum(counts.values())
    total_ce = sum(sums.values())
    out = {
        "n_tokens": total_n,
        "mean_ce": (total_ce / total_n) if total_n else float("nan"),
        "saturated": bool(total_n and total_ce / total_n > SATURATION_CE),
        "classes": {},
    }
    for c, name in CLASS_NAMES.items():
        n = counts[c]
        out["classes"][str(c)] = {
            "name": name,
            "n_tokens": n,
            "mean_ce": (sums[c] / n) if n else float("nan"),
        }
    return out


# ---------------------------------------------------------------- analysis


def contrast(stats, base):
    """Deltas of `stats` against the `full` baseline `base`, over the same position window."""
    d = {
        c: stats["classes"][str(c)]["mean_ce"] - base["classes"][str(c)]["mean_ce"]
        for c in CLASS_NAMES
    }
    d3 = d[3]
    return {
        "overall_ce": stats["mean_ce"],
        "d_overall": stats["mean_ce"] - base["mean_ce"],
        "d_class1": d[1],
        "d_class2": d[2],
        "d_class3": d[3],
        "excess": d[2] - d3,
        "ratio": (d[2] / d3) if abs(d3) > 1e-9 else float("nan"),
        "saturated": stats["saturated"],
    }


def verdict(excess: float, saturated: bool) -> str:
    if saturated:
        return "saturated, contrast discarded"
    if not np.isfinite(excess):
        return "undefined"
    if excess < 0:
        return "class 2 preserved better than class 3 (belief survives without writes)"
    if excess > 0:
        return "class 2 hurt more than class 3 (belief continuously refreshed)"
    return "no difference"


def is_deep(family: str) -> bool:
    """A per-level family that freezes ONLY levels the causal patching implicated (4 and 5)."""
    return family.startswith("frzL") and set(family[4:]) <= set("45")


def headline(results, seeds, conditions, cuts, window):
    per_cond = {}
    for name, mode, frac, frz in conditions:
        if name == BASELINE:
            continue
        rows = [results[f"seed{s}"][name][window]["contrast"] for s in seeds]
        keys = ["overall_ce", "d_overall", "d_class1", "d_class2", "d_class3", "excess", "ratio"]
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        mean["ratio_of_means"] = (
            float(mean["d_class2"] / mean["d_class3"])
            if abs(mean["d_class3"]) > 1e-9
            else float("nan")
        )
        mean["any_seed_saturated"] = any(r["saturated"] for r in rows)
        mean["verdict"] = verdict(mean["excess"], mean["any_seed_saturated"])
        mean["mode"] = mode
        mean["cut_frac"] = frac
        mean["freeze_levels"] = list(frz) if frz else None
        mean["family"] = name.split("@")[0]
        per_cond[name] = {"per_seed": rows, "mean": mean}

    fams = {}
    for name, cd in per_cond.items():
        fams.setdefault(cd["mean"]["family"], []).append(name)
    by_family = {}
    for fam, names in fams.items():
        names = sorted(names, key=lambda n: per_cond[n]["mean"]["cut_frac"])
        usable = [n for n in names if not per_cond[n]["mean"]["any_seed_saturated"]]
        signs = [np.sign(per_cond[n]["mean"]["excess"]) for n in usable]
        by_family[fam] = {
            "conditions": names,
            "freeze_levels": per_cond[names[0]]["mean"]["freeze_levels"],
            "excess_by_cut": {n: per_cond[n]["mean"]["excess"] for n in names},
            "overall_ce_by_cut": {n: per_cond[n]["mean"]["overall_ce"] for n in names},
            "saturated_by_cut": {n: per_cond[n]["mean"]["any_seed_saturated"] for n in names},
            "usable_conditions": usable,
            "all_saturated": not usable,
            "consistent_across_cuts": bool(len(set(signs)) <= 1 and signs),
            "verdict": (
                verdict(float(np.mean([per_cond[n]["mean"]["excess"] for n in usable])), False)
                if usable
                else "every cutoff saturated, contrast discarded"
            ),
        }

    deep = [f for f in by_family if is_deep(f)]
    shallow = [f for f in by_family if f.startswith("frzL") and not is_deep(f)]
    deep_sat = all(by_family[f]["all_saturated"] for f in deep) if deep else None
    shallow_sat = all(by_family[f]["all_saturated"] for f in shallow) if shallow else None
    whole_sat = by_family["frozen"]["all_saturated"] if "frozen" in by_family else None
    return {
        "window": window,
        "per_condition": per_cond,
        "by_family": by_family,
        "diagnosis": {
            "deep_families": deep,
            "shallow_families": shallow,
            "deep_freeze_all_saturated": deep_sat,
            "shallow_freeze_all_saturated": shallow_sat,
            "wholesale_freeze_all_saturated": whole_sat,
            "shallow_saturates_while_deep_does_not": bool(
                deep and shallow and shallow_sat and not deep_sat
            ),
        },
    }


# ---------------------------------------------------------------- printing


def print_window(results, seeds, conditions, h, window, cut_pos, label):
    print("\n" + "=" * 104)
    print(f"FROZEN-SIGMA ROLLOUT [{label}], mean over seeds {seeds}   (EXPLORATORY)")
    print("=" * 104)
    b = results[f"seed{seeds[0]}"][BASELINE]["all"]
    bm = float(np.mean([results[f"seed{s}"][BASELINE]["all"]["mean_ce"] for s in seeds]))
    print(
        f"  baseline `full`, all positions: overall {bm:.5f} | "
        f"c1 {b['classes']['1']['mean_ce']:.4f} | c3 {b['classes']['3']['mean_ce']:.4f} | "
        f"c2 {b['classes']['2']['mean_ce']:.4f}   (seed {seeds[0]} per-class)"
    )
    print(
        f"{'condition':<14s} {'cutPos':>7s} {'nTok c2':>8s} {'baseCE':>9s} {'overall':>9s} "
        f"{'d c1':>9s} {'d c3':>9s} {'d c2':>9s} {'excess':>9s} {'ratio':>7s}"
    )
    for name, _mode, frac, _frz in conditions:
        if name == BASELINE:
            continue
        cp = "--" if frac is None else str(cut_pos[frac])
        s0 = results[f"seed{seeds[0]}"][name][window]
        n2 = s0["classes"]["2"]["n_tokens"]
        base_ce = float(
            np.mean([results[f"seed{s}"][name][f"baseline_{window}"]["mean_ce"] for s in seeds])
        )
        m = h["per_condition"][name]["mean"]
        flag = "  SATURATED" if m["any_seed_saturated"] else ""
        print(
            f"{name:<14s} {cp:>7s} {n2:>8d} {base_ce:9.4f} {m['overall_ce']:9.4f} "
            f"{m['d_class1']:9.4f} {m['d_class3']:9.4f} {m['d_class2']:9.4f} {m['excess']:+9.4f} "
            f"{m['ratio_of_means']:7.2f}{flag}"
        )
    print("\n  c1 = empty/wall, c2 = memory-relevant object, c3 = other object (control).")
    print("  deltas are against the `full` baseline restricted to the SAME position window,")
    print("  whose overall CE in that window is the baseCE column.")
    print("  excess = d(class2) - d(class3). NEGATIVE = class 2 preserved better than class 3.")
    print(f"  nTok c2 is the class-2 token count in this window at seed {seeds[0]}.")


FAMILY_BLURB = {
    "frozen": "wholesale: every level stops updating",
    "scaled0": "wholesale, decay kept and writes stopped (see the gamma note)",
}


def print_verdict(h, gamma):
    print("\n" + "-" * 104)
    print("WHAT IT MEANS")
    print("-" * 104)
    for fam, fd in h["by_family"].items():
        lv = fd["freeze_levels"]
        blurb = FAMILY_BLURB.get(fam, f"per-level: only levels {lv} stop updating")
        print(f"\n  {fam}  ({blurb})")
        for n in fd["conditions"]:
            m = h["per_condition"][n]["mean"]
            sat = " SAT" if m["any_seed_saturated"] else "    "
            print(
                f"    {n:<14s} overall {m['overall_ce']:8.4f}{sat} excess {m['excess']:+8.4f} "
                f"nats  ->  {m['verdict']}"
            )
        print(
            f"    consistent sign across cutoffs: {fd['consistent_across_cuts']}   "
            f"family verdict: {fd['verdict']}"
        )

    dg = h["diagnosis"]
    print("\n" + "-" * 104)
    print("DIAGNOSIS")
    print("-" * 104)
    if dg["wholesale_freeze_all_saturated"]:
        print(
            "  The WHOLESALE frozen arms saturate at every cutoff (overall CE > "
            f"{SATURATION_CE} nat).\n"
            "  Read that as a fact about the intervention, NOT as a null for the belief\n"
            "  hypothesis. `sigma` is the whole associative memory, not a separable long-term\n"
            "  store: freezing all of it also removes the read of the LAST FEW tokens, which\n"
            "  carry the agent's current coordinates and the current observation window. The\n"
            "  damage to class 1 (empty/wall cells, which need no memory at all) shows that\n"
            "  directly. With every class pinned far above chance the class-2 vs class-3\n"
            "  contrast is a ceiling artefact and is discarded. This bounded result stands on\n"
            "  its own: sigma cannot be ablated as memory separately from current context by\n"
            "  freezing it wholesale."
        )
    if dg["shallow_saturates_while_deep_does_not"]:
        print(
            "\n  CONFIRMED: the SHALLOW freeze "
            f"{dg['shallow_families']} saturates while the DEEP freeze\n"
            f"  {dg['deep_families']} does not. That is exactly the predicted asymmetry, and it\n"
            "  confirms the diagnosis above: the wholesale arms failed because they also froze\n"
            "  the shallow levels that carry current context, not because sigma holds nothing.\n"
            "  The deep-freeze contrasts are therefore the informative ones; read their excess."
        )
    elif dg["deep_freeze_all_saturated"] and dg["deep_families"]:
        print(
            "\n  The DEEP freeze "
            f"{dg['deep_families']} ALSO saturates, so freezing by depth does not escape the\n"
            "  ceiling either. The contrast is discarded and the belief hypothesis is untested\n"
            "  by this design; a still narrower intervention would be needed."
        )
    elif dg["deep_families"] and not dg["deep_freeze_all_saturated"]:
        print(
            "\n  The DEEP freeze "
            f"{dg['deep_families']} stays under the {SATURATION_CE}-nat guard, so its contrast\n"
            "  is usable. Note that the shallow control did NOT saturate as predicted, so the\n"
            "  deep/shallow asymmetry is weaker than the causal patching implied; report the\n"
            "  deep excess on its own terms rather than as a confirmed dissociation."
        )
    if gamma == 1.0:
        print(
            "\n  GAMMA = 1.0: there is no decay to keep, so `scaled0` (decay kept, writes stopped)\n"
            "  and `frozen` (both stopped) are the SAME computation. Their agreement below is an\n"
            "  identity check on the code path, NOT independent evidence."
        )
    else:
        print(
            f"\n  GAMMA = {gamma}: `scaled0` keeps the decay and stops the writes, `frozen` stops\n"
            "  both, so the pair separates 'stop writing' from 'stop decaying'."
        )
    print("-" * 104)
    print("Post-hoc and exploratory. Decides nothing; cannot revise H1-H8.")


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true", help="seed 0, 20 episodes, 1 cut, no JSON")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--cuts", type=float, nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=["frozen", "scaled0"])
    ap.add_argument(
        "--freeze-sets",
        nargs="*",
        default=None,
        help="per-level freeze subsets, e.g. 4,5 5 0,1,2,3 ; pass `none` for no per-level arms",
    )
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else ([0] if args.validate else SEEDS)
    n_ep = args.episodes if args.episodes is not None else (20 if args.validate else N_EPISODES)
    cuts = args.cuts if args.cuts is not None else ([0.50] if args.validate else CUTS)
    chunk = args.chunk if not args.validate else min(args.chunk, n_ep)
    if args.freeze_sets is None:
        freeze_sets = FREEZE_SETS
    elif len(args.freeze_sets) == 1 and args.freeze_sets[0].lower() == "none":
        freeze_sets = []
    else:
        freeze_sets = [tuple(int(v) for v in g.replace(",", "")) for g in args.freeze_sets]
    for lv in freeze_sets:
        assert lv and all(0 <= v < N_LAYER for v in lv), f"bad freeze set {lv}"
    conditions = build_conditions(cuts, args.modes, freeze_sets)

    device = select_device(None)
    print(f"EXPLORATORY post-hoc frozen-sigma rollout | device={device} root={ROOT}")
    print(f"seeds={seeds} episodes={n_ep} chunk={chunk} cuts={cuts} modes={args.modes}")
    print(f"freeze_sets={freeze_sets}")
    print(f"conditions={[c[0] for c in conditions]}\n")

    d = EpisodeData(str(DATA_DIR), "probe_test")
    cls = build_classes(d)
    check_classes(d, cls)
    cls_t = cls[:, 1:]  # full index w -> target index w-1
    mask_t = d.loss_mask[1:].astype(bool)
    n_ep = min(n_ep, d.n)
    indices = np.arange(n_ep)
    cls_sub = cls_t[indices]
    T_run = d.T - 1
    cut_pos = {f: int(round(f * T_run)) for f in cuts}

    sub = cls_sub[:, mask_t]
    tot = sub.size
    print(f"token-class census over the {n_ep}-episode SUBSET (masked target positions):")
    for c, name in CLASS_NAMES.items():
        k = int((sub == c).sum())
        print(f"  class {c} {name:<24s} {k:9d}  {100 * k / tot:6.3f}%")
    n_c2 = int((sub == 2).sum())
    print(f"  -> the subset yields {n_c2} class-2 (memory-relevant) tokens in total.")
    print(f"  T_run={T_run}, cut positions: " + ", ".join(f"{int(100 * f)}%={cut_pos[f]}" for f in cuts))
    for f in cuts:
        k = int((cls_sub[:, mask_t & (np.arange(T_run) >= cut_pos[f])] == 2).sum())
        print(f"     class-2 tokens at or after the {int(100 * f)}% cut: {k}")
    print()

    results, checks = {}, {}
    gamma = None
    for seed in seeds:
        ckpt = ROOT / f"runs/study1/{RUN}/seed{seed}/ckpt.pt"
        model, _, meta = load_checkpoint(str(ckpt), device)
        model.eval()
        assert model.hcfg.n_layer == N_LAYER, f"n_layer {model.hcfg.n_layer} != {N_LAYER}"
        gamma = float(model.hcfg.decay_gamma)
        print(f"seed {seed}: step={meta['step']} val_ce={meta['val_ce']:.4f} gamma={gamma}")
        results[f"seed{seed}"] = {}
        ce_by_cond, meta_by_cond = {}, {}
        for name, mode, frac, frz in conditions:
            t0 = time.time()
            cut = None if frac is None else cut_pos[frac]
            # The per-level arms keep plasticity "full" and undo the write instead.
            m, scale = {
                None: ("full", 1.0),
                "frozen": ("frozen", 0.0),
                "scaled0": ("scaled", 0.0),
                "levelfrozen": ("full", 1.0),
            }[mode]
            ce, dg = run_condition(model, d, indices, cut, m, scale, frz, chunk, device)
            ce_by_cond[name] = ce
            meta_by_cond[name] = (cut, dg)
            # step() is called once per position PER CHUNK (it advances the whole batch at once).
            n_chunks = -(-len(indices) // chunk)
            assert dg["n_full"] + dg["n_post"] == T_run * n_chunks, f"{name}: step count wrong"
            if frz is None:
                exp_post = 0 if cut is None else (T_run - cut) * n_chunks
                assert dg["n_post"] == exp_post, f"{name}: {dg['n_post']} post-cut, {exp_post} exp"
                if m == "frozen":
                    assert dg["sigma_ok"], f"{name}: sigma moved under plasticity='frozen'"
                note = f"frozenSigmaOK={dg['sigma_ok']}"
            else:
                exp_r = (T_run - cut) * n_chunks
                assert dg["n_post"] == 0, f"{name}: plasticity should stay 'full' throughout"
                assert dg["n_restores"] == exp_r, f"{name}: {dg['n_restores']} restores, {exp_r} exp"
                assert dg["frozen_wrote"], (
                    f"{name}: the frozen levels never wrote, so the restore is a no-op"
                )
                assert dg["frozen_identical_after_restore"], (
                    f"{name}: frozen levels not bit-identical to the cut-time snapshot"
                )
                assert dg["unfrozen_changed"], (
                    f"{name}: an unfrozen level did not change, so the freeze is not selective"
                )
                assert dg["frozen_identical_at_end"], (
                    f"{name}: frozen levels drifted by the final position"
                )
                note = (
                    f"freeze={list(frz)} restores={dg['n_restores']} "
                    f"bitIdent={dg['frozen_identical_at_end']} "
                    f"unfrozenMoved={dg['unfrozen_changed']}"
                )
            st = class_stats(ce, cls_sub, mask_t, 0)
            print(
                f"  {name:<14s} cut={str(cut):>5s} overall {st['mean_ce']:8.5f} | "
                f"c1 {st['classes']['1']['mean_ce']:7.4f} | "
                f"c3 {st['classes']['3']['mean_ce']:7.4f} | "
                f"c2 {st['classes']['2']['mean_ce']:7.4f} | "
                f"{note} [{time.time() - t0:.1f}s]"
            )

        # Every delta is taken against the `full` baseline RESTRICTED TO THE SAME window, so the
        # untouched prefix never contaminates the post-cut contrast.
        base_ce = ce_by_cond[BASELINE]
        for name, _mode, _frac, frz in conditions:
            cut, dg = meta_by_cond[name]
            lo = cut or 0
            entry = {
                "cut_pos": cut,
                "freeze_levels": list(frz) if frz else None,
                "diagnostics": dg,
                "all": class_stats(ce_by_cond[name], cls_sub, mask_t, 0),
                "post_cut": class_stats(ce_by_cond[name], cls_sub, mask_t, lo),
                "baseline_all": class_stats(base_ce, cls_sub, mask_t, 0),
                "baseline_post_cut": class_stats(base_ce, cls_sub, mask_t, lo),
            }
            if name != BASELINE:
                for w in ("all", "post_cut"):
                    entry[w]["contrast"] = contrast(entry[w], entry[f"baseline_{w}"])
            results[f"seed{seed}"][name] = entry
        for name, _m, _f, _l in conditions:
            if name == BASELINE:
                continue
            assert not np.array_equal(ce_by_cond[name], base_ce), (
                f"seed {seed}: {name} per-token CE identical to `full` -- intervention inert"
            )
        # A per-level freeze must also differ from the wholesale freeze at the same cutoff.
        for name, _m, f, lv in conditions:
            w = cond_name("frozen", f)
            if lv is None or w not in ce_by_cond:
                continue
            assert not np.array_equal(ce_by_cond[name], ce_by_cond[w]), (
                f"seed {seed}: {name} identical to {w} -- the level subset had no effect"
            )

        # -- correctness check: recurrent `full` vs the parallel path on the same episodes.
        ce_par = run_parallel(model, d, indices, chunk, device)
        par = class_stats(ce_par, cls_sub, mask_t, 0)
        rec = results[f"seed{seed}"][BASELINE]["all"]
        diffs = {
            "overall": abs(rec["mean_ce"] - par["mean_ce"]),
            **{
                f"class{c}": abs(
                    rec["classes"][str(c)]["mean_ce"] - par["classes"][str(c)]["mean_ce"]
                )
                for c in CLASS_NAMES
            },
        }
        rels = {
            k: v / max(abs(par["mean_ce"] if k == "overall" else
                          par["classes"][k[-1]]["mean_ce"]), 1e-12)
            for k, v in diffs.items()
        }
        ok = all(diffs[k] <= AGREE_ABS or rels[k] <= AGREE_REL for k in diffs)
        checks[f"seed{seed}"] = {
            "parallel": {k: v for k, v in par.items() if k != "classes"},
            "parallel_classes": {c: par["classes"][c]["mean_ce"] for c in par["classes"]},
            "recurrent_classes": {c: rec["classes"][c]["mean_ce"] for c in rec["classes"]},
            "abs_diff": diffs,
            "rel_diff": rels,
            "agrees": bool(ok),
        }
        print(f"  CORRECTNESS CHECK seed {seed}: recurrent `full` vs parallel, same {n_ep} episodes")
        print(
            f"    overall  recurrent {rec['mean_ce']:.6f}  parallel {par['mean_ce']:.6f}  "
            f"absdiff {diffs['overall']:.2e}"
        )
        for c in (1, 3, 2):
            print(
                f"    class {c}  recurrent {rec['classes'][str(c)]['mean_ce']:.6f}  "
                f"parallel {par['classes'][str(c)]['mean_ce']:.6f}  "
                f"absdiff {diffs[f'class{c}']:.2e}"
            )
        print(f"    -> {'AGREES' if ok else 'DIVERGES -- the recurrent driver is WRONG'}")
        if not ok:
            print(
                "    The rest of this report is NOT trustworthy: the recurrent `full` condition "
                "must reproduce\n    the parallel path before any frozen contrast means anything."
            )

        if gamma == 1.0:
            for f in cuts:
                a, b = cond_name("frozen", f), cond_name("scaled0", f)
                if a in ce_by_cond and b in ce_by_cond:
                    assert np.array_equal(ce_by_cond[a], ce_by_cond[b]), (
                        f"gamma=1.0 but {a} and {b} differ; the plasticity gating changed"
                    )
        del model, ce_by_cond
        release_memory(device)
        print()

    h_post = headline(results, seeds, conditions, cuts, "post_cut")
    h_all = headline(results, seeds, conditions, cuts, "all")

    print_window(results, seeds, conditions, h_post, "post_cut", cut_pos,
                 "POST-CUT positions only, the headline")
    print_window(results, seeds, conditions, h_all, "all", cut_pos,
                 "ALL positions, diluted by the untouched prefix")
    print_verdict(h_post, gamma)

    if not args.validate:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": "EXPLORATORY, post-hoc, NOT preregistered. Cannot revise H1-H8.",
            "gamma_note": (
                "At decay_gamma == 1.0 there is no decay, so `scaled0` (decay kept, writes "
                "stopped) and `frozen` (both stopped) are the same computation and produce "
                "identical numbers by construction. Their agreement is an identity check on the "
                "code path, not independent evidence."
            ),
            "window_note": (
                "post_cut restricts to masked positions at or after the cutoff, the only region "
                "where the intervention is active, and is the headline. `all` includes the "
                "untouched prefix and merely dilutes the same effect."
            ),
            "reading_note": (
                "excess = d(class2) - d(class3) against the `full` baseline. Negative means "
                "freezing preserves memory-relevant predictions relatively better than the "
                "matched no-memory control, consistent with sigma holding a belief that survives "
                "without further writes. Positive means the belief is continuously refreshed "
                "rather than stored."
            ),
            "run": RUN,
            "split": "probe_test",
            "n_episodes": int(n_ep),
            "n_class2_tokens_in_subset": n_c2,
            "chunk": int(chunk),
            "T_run": int(T_run),
            "cuts": cuts,
            "cut_positions": {str(f): cut_pos[f] for f in cuts},
            "freeze_sets": [list(lv) for lv in freeze_sets],
            "decay_gamma": gamma,
            "saturation_ce_threshold": SATURATION_CE,
            "conditions": [c[0] for c in conditions],
            "per_level_note": (
                "The per-level arms keep plasticity 'full' and undo the write: the frozen levels' "
                "sigma slices are restored to their cut-time value after every step from the cut "
                "onward, which is exactly equivalent to those levels never updating. hbwm/ is not "
                "touched. Bit-identity of the frozen slice, a real write before each restore, and "
                "movement of an unfrozen level are all asserted."
            ),
            "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
            "recurrent_vs_parallel_check": checks,
            "per_seed": results,
            "headline_post_cut": h_post,
            "headline_all": h_all,
        }
        OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
