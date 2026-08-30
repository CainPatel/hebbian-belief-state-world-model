"""Per-HEAD causal patching of BDH's associative read (`yKV`), post-hoc and EXPLORATORY.

NOT PREREGISTERED. This script cannot revise H1-H8, or any other preregistered decision from
Study 1 or Study 2. It decides nothing. It exists so these exploratory numbers are reproducible.

The question
------------
`analysis/posthoc/ykv_causal_patch.py` patched one LEVEL at a time and found the memory-specific
damage concentrated in the deep levels: mean over seeds 0/1/2, against an intact overall CE of
0.0246, patching L0 costs 2.77 nats overall (saturated), L1 1.21 (saturated), L2 0.50, L3 0.17,
L4 0.089, L5 0.052, while the SELECTIVITY (class-2 delta over class-3 delta) runs the other way:
L2 1.15, L3 1.13, L4 2.26, L5 5.35. Patching L5 leaves overall prediction nearly intact yet costs
about 2.59 nats on memory-relevant cells alone.

Each level has `n_head = 4` heads. This script localizes one step further: is the memory-specific
effect carried by ONE head, SHARED across the four, or does it need them JOINTLY?

The intervention
----------------
Identical machinery to the level script, reused rather than reimplemented: its token-class
labelling (`build_classes` / `check_classes`), its `AttendInterceptor` monkeypatch of the BOUND
`_attend` and `forward` on the loaded instance, its per-condition summary and its saturation guard.
`hbwm/` is never touched.

`_attend` returns `[B, nh, T, D]`. A single-head condition clones that tensor and replaces ONLY
`out[:, h]` with `out[:, h].roll(1, dims=0)`: at one level, one head reads another episode's
associative read. Same marginal distribution, same norms, wrong content. The other three heads and
all other levels run untouched.

`self.ln` is `nn.LayerNorm(D, elementwise_affine=False, bias=False)` and is SCALE-INVARIANT, so
rescaling the read is a no-op and a useless intervention. The roll changes CONTENT, not scale.

Because the batch is processed in chunks, the roll is WITHIN a chunk. Every episode still receives
another episode's read; only the specific donor depends on the chunking.

Conditions: `intact`; `patched_L4H0..H3` and `patched_L5H0..H3` (the eight single-head arms);
`patched_L4` and `patched_L5` (the whole-level arms, as the reference each level's heads are
compared against). Seeds 0, 1, 2, the full 2,000 `probe_test` episodes.

What is measured
----------------
Per-token cross-entropy over `probe_test`, aggregated into the four classes the loss-attribution
pass used (0 non-window, 1 window empty/wall, 3 object needing NO memory, 2 object RETURNING after
an absence to the cell it was last seen at). Per condition: overall CE, the class 1/2/3 deltas
against intact, the excess `d(class2) - d(class3)` and the ratio `d(class2) / d(class3)`, exactly
as the level script reports them.

Then, per level, ADDITIVITY: is `sum over h of excess(L, h)` approximately `excess(L)`?

  approximately additive  the four heads contribute roughly independent, separable damage;
                          combined with a concentrated per-head profile this is "one head does it",
                          and with a flat profile it is "the heads share it".
  sub-additive            the whole-level patch costs LESS than the sum of its parts, i.e. the
                          heads are partly redundant, each able to stand in for the others.
  super-additive          the whole-level patch costs MORE than the sum of its parts, i.e. the
                          effect needs the heads JOINTLY and no single head reveals it.

SATURATION -- do not read a saturated condition as a null
---------------------------------------------------------
Any condition whose overall CE exceeds SATURATION_CE (1.0 nat, against an intact 0.0246) is in the
regime where both object classes are pinned near a confidently-wrong floor, so the class-2 vs
class-3 contrast carries no information. Such conditions are FLAGGED and their contrast is excluded
from the additivity accounting rather than reported as a null. The deep-level arms measured by the
level script sit far below that threshold, so no head condition is expected to trip it; the guard is
kept because an unexpected trip would invalidate the arm, not the hypothesis.

Usage
-----
    uv run python analysis/posthoc/ykv_head_patch.py                    # full run, writes JSON
    uv run python analysis/posthoc/ykv_head_patch.py --validate         # seed 0, 100 eps, L5H0
    uv run python analysis/posthoc/ykv_head_patch.py --levels 5

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
from hbwm.train import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ykv_causal_patch import (  # noqa: E402
    CHUNK,
    CLASS_NAMES,
    DATA_DIR,
    N_LAYER,
    ROOT,
    RUN,
    SATURATION_CE,
    SEEDS,
    AttendInterceptor,
    build_classes,
    check_classes,
    summarize,
)

OUT_PATH = ROOT / "runs/study1/results2/posthoc_ykv_head_patch.json"

N_EPISODES = 2000
LEVELS = [4, 5]
N_HEAD = 4

# Sum-of-heads over whole-level excess inside this band counts as "approximately additive".
ADDITIVITY_LO, ADDITIVITY_HI = 0.80, 1.25


# ---------------------------------------------------------------- conditions


def head_cond(level: int, head: int) -> str:
    return f"patched_L{level}H{head}"


def level_cond(level: int) -> str:
    return f"patched_L{level}"


def parse_condition(condition: str):
    """Returns (level, head) with head None for a whole-level arm, or (None, None) for intact."""
    if condition == "intact":
        return None, None
    body = condition[len("patched_L") :]
    if "H" in body:
        lv, hd = body.split("H")
        return int(lv), int(hd)
    return int(body), None


def transform_for(condition: str):
    """Returns f(level, out) -> out, with `out` shaped [B, nh, T, D]."""
    target, head = parse_condition(condition)
    if target is None:
        return lambda level, out: out
    if head is None:
        return lambda level, out: out.roll(1, dims=0) if level == target else out

    def one_head(level, out):
        if level != target:
            return out
        assert head < out.size(1), f"head {head} out of range for nh={out.size(1)}"
        new = out.clone()
        new[:, head] = out[:, head].roll(1, dims=0)
        return new

    return one_head


@contextmanager
def intervened(model, condition: str):
    """Install the reused `AttendInterceptor` for the duration. `hbwm/` is never touched."""
    ic = AttendInterceptor(model, transform_for(condition))
    model._attend = ic.attend  # instance attributes shadow the class methods
    model.forward = ic.forward
    try:
        yield ic
    finally:
        del model._attend
        del model.forward
    assert model._attend.__func__ is ic.orig_attend.__func__, "failed to restore _attend"
    assert model.forward.__func__ is ic.orig_forward.__func__, "failed to restore forward"


# ---------------------------------------------------------------- measurement


@torch.no_grad()
def run_condition(model, d, cls_t, mask_t, indices, condition, chunk, device):
    sums = {c: 0.0 for c in CLASS_NAMES}
    counts = {c: 0 for c in CLASS_NAMES}
    with intervened(model, condition) as ic:
        for start in range(0, len(indices), chunk):
            idx = indices[start : start + chunk]
            assert len(idx) >= 2, "chunk of 1 makes roll(1, dims=0) a no-op"
            x, y, _ = d.batch_at(idx, device)
            logits, _ = model(x)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).view_as(y)
            ce_np = ce.float().cpu().numpy()
            cls_chunk = cls_t[idx]
            for c in CLASS_NAMES:
                m = (cls_chunk == c) & mask_t[None, :]
                counts[c] += int(m.sum())
                sums[c] += float(ce_np[m].sum())
        n_forwards, patched_levels = ic.forwards, sorted(ic.patched_levels)
    target, _ = parse_condition(condition)
    expected = [] if target is None else [target]
    assert patched_levels == expected, (
        f"{condition}: transformed levels {patched_levels}, expected {expected}"
    )
    total_n = sum(counts.values())
    total_ce = sum(sums.values())
    return counts, sums, total_n, total_ce, n_forwards, patched_levels


# ---------------------------------------------------------------- analysis


def _ce(results, seed, cond, c):
    return results[f"seed{seed}"][cond]["classes"][str(c)]["mean_ce"]


def deltas(results, seed, cond):
    """CE increase over intact, per class, for one (seed, condition)."""
    return {c: _ce(results, seed, cond, c) - _ce(results, seed, "intact", c) for c in CLASS_NAMES}


def condition_stats(results, seeds, cond):
    per_seed = []
    for s in seeds:
        d = deltas(results, s, cond)
        d3 = d[3]
        per_seed.append(
            {
                "seed": s,
                "overall_ce": results[f"seed{s}"][cond]["mean_ce"],
                "saturated": results[f"seed{s}"][cond]["saturated"],
                "d_class1": d[1],
                "d_class2": d[2],
                "d_class3": d[3],
                "excess": d[2] - d3,
                "ratio": (d[2] / d3) if abs(d3) > 1e-9 else float("nan"),
            }
        )
    keys = ["overall_ce", "d_class1", "d_class2", "d_class3", "excess", "ratio"]
    mean = {k: float(np.nanmean([p[k] for p in per_seed])) for k in keys}
    mean["ratio_of_means"] = (
        float(mean["d_class2"] / mean["d_class3"]) if abs(mean["d_class3"]) > 1e-9 else float("nan")
    )
    mean["any_seed_saturated"] = any(p["saturated"] for p in per_seed)
    return {"per_seed": per_seed, "mean": mean}


def classify_additivity(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "undefined"
    if ratio < ADDITIVITY_LO:
        return "sub-additive"
    if ratio > ADDITIVITY_HI:
        return "super-additive"
    return "approximately additive"


def additivity(per_cond, levels, heads):
    """Per level: sum of single-head excesses vs the whole-level excess, mean over seeds."""
    out = {}
    for lv in levels:
        hconds = [head_cond(lv, h) for h in heads if head_cond(lv, h) in per_cond]
        lcond = level_cond(lv)
        if lcond not in per_cond or not hconds:
            continue
        usable = [c for c in hconds if not per_cond[c]["mean"]["any_seed_saturated"]]
        excluded = [c for c in hconds if c not in usable]
        head_ex = {c: per_cond[c]["mean"]["excess"] for c in usable}
        s = float(sum(head_ex.values()))
        whole = per_cond[lcond]["mean"]["excess"]
        ratio = (s / whole) if abs(whole) > 1e-9 else float("nan")
        top = max(head_ex.values()) if head_ex else float("nan")
        pos = sum(v for v in head_ex.values() if v > 0)
        out[str(lv)] = {
            "heads_used": usable,
            "heads_excluded_saturated": excluded,
            "level_saturated": per_cond[lcond]["mean"]["any_seed_saturated"],
            "per_head_excess": head_ex,
            "sum_of_head_excess": s,
            "whole_level_excess": float(whole),
            "sum_over_whole": float(ratio),
            "verdict": classify_additivity(ratio),
            "top_head": (
                max(head_ex, key=head_ex.get) if head_ex else None
            ),
            "top_head_share_of_positive_sum": float(top / pos) if pos > 1e-9 else float("nan"),
        }
    return out


def headline(results, seeds, conditions, levels, heads):
    per_cond = {c: condition_stats(results, seeds, c) for c in conditions if c != "intact"}
    add = additivity(per_cond, levels, heads)
    hconds = [c for c in per_cond if parse_condition(c)[1] is not None]
    usable = [c for c in hconds if not per_cond[c]["mean"]["any_seed_saturated"]] or hconds
    best = max(usable, key=lambda c: per_cond[c]["mean"]["excess"]) if usable else None
    return {
        "saturation_ce_threshold": SATURATION_CE,
        "saturated_conditions": [c for c in per_cond if per_cond[c]["mean"]["any_seed_saturated"]],
        "per_condition": per_cond,
        "additivity": add,
        "most_selective_head": best,
        "additivity_band": [ADDITIVITY_LO, ADDITIVITY_HI],
    }


# ---------------------------------------------------------------- printing


def _row(name, m, note=""):
    return (
        f"{name:<14s} {m['overall_ce']:9.4f} {m['d_class1']:8.4f} {m['d_class3']:9.4f} "
        f"{m['d_class2']:9.4f} {m['excess']:+9.4f} {m['ratio_of_means']:7.2f}  {note}"
    )


def print_summary(results, seeds, h, levels, heads):
    print("=" * 104)
    print(f"PER-HEAD yKV PATCHING, mean over seeds {seeds}   (EXPLORATORY, not preregistered)")
    print("=" * 104)
    print(
        f"{'condition':<14s} {'overallCE':>9s} {'d c1':>8s} {'d c3':>9s} {'d c2':>9s} "
        f"{'excess':>9s} {'ratio':>7s}  note"
    )
    intact_ce = float(np.mean([results[f"seed{s}"]["intact"]["mean_ce"] for s in seeds]))
    print(
        f"{'intact':<14s} {intact_ce:9.4f} {0.0:8.4f} {0.0:9.4f} {0.0:9.4f} {0.0:9.4f} "
        f"{'--':>7s}  reference"
    )
    for lv in levels:
        for hd in heads:
            c = head_cond(lv, hd)
            if c not in h["per_condition"]:
                continue
            m = h["per_condition"][c]["mean"]
            note = "SATURATED - contrast discarded" if m["any_seed_saturated"] else ""
            print(_row(c, m, note))
        c = level_cond(lv)
        if c in h["per_condition"]:
            m = h["per_condition"][c]["mean"]
            note = "whole level (all 4 heads), reference"
            if m["any_seed_saturated"]:
                note = "SATURATED - contrast discarded; " + note
            print(_row(c, m, note))
    print("\n  c1 = empty/wall, c2 = memory-relevant object, c3 = other object (control).")
    print("  excess = d(class2) - d(class3), in nats. ratio = mean d(c2) / mean d(c3).")

    print("\n" + "-" * 104)
    print("PER-SEED excess (nats) by patched (level, head)")
    print("-" * 104)
    hdr = f"{'seed':<6s}" + "".join(
        f"{f'L{lv}H{hd}':>12s}" for lv in levels for hd in heads
    ) + "".join(f"{f'L{lv}all':>12s}" for lv in levels)
    print(hdr)
    order = [head_cond(lv, hd) for lv in levels for hd in heads] + [level_cond(lv) for lv in levels]
    for s in seeds:
        row = f"{s:<6d}"
        for c in order:
            if c not in h["per_condition"]:
                continue
            d = deltas(results, s, c)
            flag = "*" if results[f"seed{s}"][c]["saturated"] else " "
            row += f"{d[2] - d[3]:>+11.4f}{flag}"
        print(row)
    print(f"  * = overall CE > {SATURATION_CE} nat for that seed/condition; contrast unreliable.")

    print("\n" + "-" * 104)
    print("ADDITIVITY: is the whole-level effect the sum of its heads?")
    print("-" * 104)
    for lv in levels:
        a = h["additivity"].get(str(lv))
        if a is None:
            continue
        parts = "  ".join(
            f"H{hd}={a['per_head_excess'].get(head_cond(lv, hd), float('nan')):+.4f}"
            for hd in heads
        )
        print(f"  L{lv}  per-head excess: {parts}")
        print(
            f"       sum of heads {a['sum_of_head_excess']:+.4f}   whole level "
            f"{a['whole_level_excess']:+.4f}   sum/whole = {a['sum_over_whole']:.2f}"
            f"   -> {a['verdict'].upper()}"
        )
        share = a["top_head_share_of_positive_sum"]
        print(
            f"       most selective head {a['top_head']}, carrying "
            f"{100 * share:.1f}% of the positive per-head excess at this level."
        )
        if a["heads_excluded_saturated"]:
            print(f"       EXCLUDED as saturated: {a['heads_excluded_saturated']}")
    print(
        f"\n  sum/whole in [{ADDITIVITY_LO}, {ADDITIVITY_HI}] = approximately additive (separable "
        "heads);\n  below = sub-additive (heads partly redundant, each can stand in for the "
        "others);\n  above = super-additive (the effect needs the heads jointly)."
    )
    print("-" * 104)
    print("Post-hoc and exploratory. Decides nothing; cannot revise H1-H8.")


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true", help="seed 0, 100 episodes, L5H0, no JSON")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--heads", type=int, nargs="*", default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else ([0] if args.validate else SEEDS)
    n_ep = args.episodes if args.episodes is not None else (100 if args.validate else N_EPISODES)
    levels = args.levels if args.levels is not None else LEVELS
    heads = args.heads if args.heads is not None else list(range(N_HEAD))
    if args.validate and args.levels is None and args.heads is None:
        levels, heads = [5], [0]

    conditions = (
        ["intact"]
        + [head_cond(lv, hd) for lv in levels for hd in heads]
        + [level_cond(lv) for lv in levels]
    )

    device = select_device(None)
    print(f"EXPLORATORY post-hoc yKV per-HEAD patch | device={device} root={ROOT}")
    print(f"seeds={seeds} episodes={n_ep} chunk={args.chunk} levels={levels} heads={heads}")
    print(f"conditions={conditions}\n")

    d = EpisodeData(str(DATA_DIR), "probe_test")
    cls = build_classes(d)
    check_classes(d, cls)
    cls_t = cls[:, 1:]  # full index w -> target index w-1
    mask_t = d.loss_mask[1:].astype(bool)
    n_ep = min(n_ep, d.n)
    indices = np.arange(n_ep)

    sub = cls_t[indices][:, mask_t]
    tot = sub.size
    print(f"token-class census over {n_ep} episodes (masked target positions):")
    for c, name in CLASS_NAMES.items():
        k = int((sub == c).sum())
        print(f"  class {c} {name:<24s} {k:9d}  {100 * k / tot:6.3f}%")
    print("  expected: class 2 ~0.55%, class 3 ~2.1%\n")

    results = {}
    for seed in seeds:
        ckpt = ROOT / f"runs/study1/{RUN}/seed{seed}/ckpt.pt"
        model, _, meta = load_checkpoint(str(ckpt), device)
        model.eval()
        assert model.hcfg.n_layer == N_LAYER, f"n_layer {model.hcfg.n_layer} != {N_LAYER}"
        assert model.hcfg.n_head == N_HEAD, f"n_head {model.hcfg.n_head} != {N_HEAD}"
        assert max(levels) < N_LAYER and max(heads) < N_HEAD, "level/head out of range"
        print(f"seed {seed}: step={meta['step']} val_ce={meta['val_ce']:.4f}")
        results[f"seed{seed}"] = {}
        for cond in conditions:
            t0 = time.time()
            counts, sums, n, ce, n_fwd, lv = run_condition(
                model, d, cls_t, mask_t, indices, cond, args.chunk, device
            )
            s = summarize(counts, sums, n, ce, lv)
            results[f"seed{seed}"][cond] = s
            c1, c2, c3 = (s["classes"][k] for k in ("1", "2", "3"))
            flag = " SAT" if s["saturated"] else "    "
            print(
                f"  {cond:<14s} overall {s['mean_ce']:8.4f}{flag} | c1 {c1['mean_ce']:7.4f} | "
                f"c3 {c3['mean_ce']:7.4f} | c2 {c2['mean_ce']:7.4f} | "
                f"lv={lv} fwd={n_fwd} [{time.time() - t0:.1f}s]"
            )
        ce_i = results[f"seed{seed}"]["intact"]["mean_ce"]
        for cond in conditions:
            if cond == "intact":
                continue
            assert abs(results[f"seed{seed}"][cond]["mean_ce"] - ce_i) > 1e-9, (
                f"seed {seed}: {cond} overall CE identical to intact -- monkeypatch inert"
            )
        del model
        release_memory(device)
        print()

    h = headline(results, seeds, conditions, levels, heads)
    print_summary(results, seeds, h, levels, heads)

    if not args.validate:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": "EXPLORATORY, post-hoc, NOT preregistered. Cannot revise H1-H8.",
            "saturation_note": (
                "Any condition whose overall CE exceeds the threshold is in the regime where both "
                "object classes are pinned near a confidently-wrong floor, so the class-2 vs "
                "class-3 contrast carries no information. Such a condition is flagged and its "
                "contrast is excluded from the additivity accounting; it is NOT a null result."
            ),
            "additivity_note": (
                "sum/whole near 1 means the four heads contribute separable damage; below the band "
                "means sub-additive (heads partly redundant); above means super-additive (the "
                "effect needs the heads jointly). Read it together with the per-head profile: "
                "additive plus concentrated is 'one head does it', additive plus flat is 'the "
                "heads share it'."
            ),
            "run": RUN,
            "split": "probe_test",
            "n_episodes": int(n_ep),
            "chunk": int(args.chunk),
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "levels": levels,
            "heads": heads,
            "conditions": conditions,
            "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
            "per_seed": results,
            "headline": h,
        }
        OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
