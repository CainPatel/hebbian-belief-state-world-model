"""Causal activation patching of BDH's associative read (`yKV`), post-hoc and EXPLORATORY.

NOT PREREGISTERED. This script cannot revise H5-H8, or any other preregistered decision from
Study 1 or Study 2. It decides nothing. It exists so these exploratory numbers are reproducible.

The question
------------
Studies 1 and 2 found that BDH's `sigma` is not a linearly or bilinearly readable belief state
(best probe 0.159 vs an LSTM's 0.146 on the same protocol). A post-hoc loss-attribution pass
nonetheless found BDH roughly twice as good as the LSTM at exactly the tokens that require
remembering an out-of-view object (mean CE 0.683 vs 1.356), so the information appears to be USED
without being linearly decodable. Probes are correlational and cannot settle that. This is the
causal test: if we corrupt the associative read, does the memory-relevant loss move?

The intervention
----------------
`HBWMCore.forward` computes, per level, `yKV = self.ln(self._attend(x_sparse, x))`. We monkeypatch
the BOUND METHODS `model._attend` and `model.forward` on the loaded instance and transform
`_attend`'s return value. `core.py` is never edited, and the class is never touched -- only the one
live object. `forward` is wrapped solely to reset the per-forward level counter and to assert that
`_attend` was invoked exactly `n_layer` times, which is what makes the level attribution sound.

`self.ln` is `nn.LayerNorm(D, elementwise_affine=False, bias=False)` and is therefore
SCALE-INVARIANT: multiplying `_attend`'s output by a constant is a no-op and a useless
intervention. The interventions here change CONTENT, not scale:

  intact       identity transform (still intercepted, so the call-count assert runs).
  patched_Lk   `out.roll(1, dims=0)` at level k ONLY; the other five levels run untouched.
               Each episode receives a DIFFERENT episode's associative read at that one level.
               Same marginal distribution, same norms, wrong content. These are the informative
               arms.
  patched      the same roll at EVERY level. Reference point only -- see the saturation note.
  zeroed       `torch.zeros_like(out)` at every level (post-LayerNorm this stays zero, since
               LayerNorm's eps keeps 0/sqrt(eps) finite). Reference point only.

Because the batch is processed in chunks, the roll is WITHIN a chunk. Every episode still receives
another episode's read; only the specific donor depends on the chunking.

SATURATION NOTE -- do not read the all-levels conditions as a null
------------------------------------------------------------------
The first version of this analysis ran only the all-levels `patched` and `zeroed` arms. Those are
catastrophic, not selective: `patched` takes overall CE from 0.024 to ~4.39, a ~180x blow-up, and
lifts even trivially-predictable empty/wall cells from 0.006 to ~3.00. At that damage level BOTH
object classes are pinned near a ~11-nat floor (worse than the 3.53 nats of a uniform distribution
over the 34-token vocabulary, i.e. confidently wrong), so the class-2 vs class-3 contrast is
SATURATED and carries no information. The all-levels ratio of ~0.94x is therefore an artefact of
that ceiling and is NOT evidence that the read is memory-agnostic. It is retained here only as the
saturation reference. The per-level arms exist to get out of that regime; any per-level condition
whose overall CE exceeds SATURATION_CE is flagged and should be discarded for the same reason.

What is measured
----------------
Per-token cross-entropy over the `probe_test` split, aggregated into the same four token classes
the loss-attribution pass used:

  0  non-window token (agent X/Y coordinates, and anything outside an observation window)
  1  window cell that is empty or wall
  3  window cell holding an object that needs NO memory (visible last step, or first sighting,
     or it moved while it was out of view, so the remembered cell is wrong anyway)
  2  window cell holding an object RETURNING after an absence to the SAME cell it was last seen
     at -- the memory-relevant class

Headline: per level, the SELECTIVITY of the damage,

    excess = (class-2 mean CE increase) - (class-3 mean CE increase)

Class 3 is the natural control: object cells of the same surface form that need no memory. A level
that carries memory of absent objects should damage class 2 MORE than class 3 when patched.

Finally the most selective level is compared against the levels Study 2's sigma probe chose as
best-on-validation (seed 0: L3 and L4; seed 1: L3; seed 2: L4). Agreement would be a convergence
between a correlational and a causal method; disagreement would mean the probe was reading a level
that is not the one doing the work. Both are reported as they fall.

Scope: `bdh_g100_lr0.003` seeds 0/1/2 (the intervention only exists for BDH), the parallel
`model(tokens[:, :-1])` path under `torch.no_grad()`.

Usage
-----
    uv run python analysis/posthoc/ykv_causal_patch.py                  # full run, writes JSON
    uv run python analysis/posthoc/ykv_causal_patch.py --validate       # seed 0, 100 eps, no JSON
    uv run python analysis/posthoc/ykv_causal_patch.py --episodes 1000

`HBWM_ROOT` overrides the artifact root (must contain `runs/` and `data/`); it defaults to the
Study 1 worktree. Read-only on checkpoints and data; the single JSON output is the only write.
"""

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hbwm.device import release_memory, select_device
from hbwm.envs import tokenizer as tk
from hbwm.envs.dataset import EpisodeData
from hbwm.train import load_checkpoint

_REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HBWM_ROOT", _REPO / ".claude/worktrees/study1-impl"))
RUN = "bdh_g100_lr0.003"
DATA_DIR = ROOT / "data/grid9"
OUT_PATH = ROOT / "runs/study1/results2/posthoc_ykv_causal.json"

SEEDS = [0, 1, 2]
N_LAYER = 6
LEVEL_CONDITIONS = [f"patched_L{k}" for k in range(N_LAYER)]
CONDITIONS = ["intact"] + LEVEL_CONDITIONS + ["patched", "zeroed"]
N_EPISODES = 2000
CHUNK = 25

# Overall CE above this is treated as the saturation regime: the class-2/class-3 contrast is
# unreliable there because both are pinned near the floor. Intact overall CE is ~0.024.
SATURATION_CE = 1.0

# Levels Study 2's sigma probe selected as best-on-validation, per seed, from
# runs/study1/results/table.json: sigma_full {0:3, 1:3, 2:4}, sigma_rownorm {0:4, 1:3, 2:4}.
# Nothing here recomputes them; they are quoted for the convergence comparison only.
STUDY2_LEVELS = {0: [3, 4], 1: [3], 2: [4]}

CLASS_NAMES = {
    0: "non_window",
    1: "window_empty_or_wall",
    2: "object_memory_relevant",
    3: "object_no_memory",
}


# ---------------------------------------------------------------- token classes


def build_classes(d: EpisodeData) -> np.ndarray:
    """Per full-token-index class label, [n, T]. Reproduces the loss-attribution labelling."""
    n, T = d.tokens.shape
    L, G = d.L, d.G

    visible = d.visible.astype(bool)  # [n, L+1, n_obj]
    cells = d.obj_pos[..., 1] * G + d.obj_pos[..., 0]  # [n, L+1, n_obj]

    prev_visible = np.zeros_like(visible)
    prev_visible[:, 1:] = visible[:, :-1]
    seen_before = (np.cumsum(visible, axis=1) - visible) > 0
    returning = visible & seen_before & ~prev_visible

    t_idx = np.arange(L + 1)[None, :, None]
    last_t = np.maximum.accumulate(np.where(prev_visible, t_idx, -1), axis=1)
    last_cell = np.take_along_axis(cells, np.clip(last_t, 0, None), axis=1)
    memory = returning & (last_cell == cells) & (last_t >= 0)

    # window slot of a visible object: (dy+1)*3 + (dx+1), matching GridWorld.window()'s
    # `for dy in (-1,0,1) for dx in (-1,0,1)` ordering, with d = obj_pos - agent_pos.
    delta = d.obj_pos - d.agent_pos[:, :, None, :]  # [n, L+1, n_obj, 2]
    slot = (delta[..., 1] + 1) * 3 + (delta[..., 0] + 1)
    # slot s of observation t is full-token index obs_positions[t] - 8 + s
    pos = tk.obs_positions(L)[None, :, None] - 8 + slot  # [n, L+1, n_obj]

    cls = np.zeros((n, T), dtype=np.int8)
    cls[:, d.window_mask] = 1
    ep, ti, ok = np.nonzero(visible)
    cls[ep, pos[ep, ti, ok]] = 3
    ep, ti, ok = np.nonzero(memory)
    cls[ep, pos[ep, ti, ok]] = 2
    return cls


def check_classes(d: EpisodeData, cls: np.ndarray) -> None:
    """The positions we called object cells must actually hold object tokens, and vice versa."""
    obj_tok = (d.tokens >= tk.OBJ_BASE) & (d.tokens < tk.VOCAB_SIZE)
    called_obj = (cls == 2) | (cls == 3)
    assert np.array_equal(called_obj, obj_tok & d.window_mask[None, :]), (
        "class 2/3 positions disagree with the object tokens actually present"
    )
    non_obj_window = cls == 1
    assert bool(
        ((d.tokens[non_obj_window] == tk.EMPTY_TOK) | (d.tokens[non_obj_window] == tk.WALL_TOK)).all()
    ), "class 1 positions are not all EMPTY/WALL"


# ---------------------------------------------------------------- intervention


def transform_for(condition: str):
    """Returns f(level, out) -> out. Level index is 0-based within one forward pass."""
    if condition == "intact":
        return lambda level, out: out
    if condition == "patched":
        return lambda level, out: out.roll(1, dims=0)
    if condition == "zeroed":
        return lambda level, out: torch.zeros_like(out)
    if condition.startswith("patched_L"):
        target = int(condition[len("patched_L") :])
        return lambda level, out: out.roll(1, dims=0) if level == target else out
    raise ValueError(f"unknown condition {condition!r}")


class AttendInterceptor:
    """Level-aware wrapper around the bound `_attend`, with a counter reset by each `forward`."""

    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.n_layer = model.hcfg.n_layer
        self.orig_attend = model._attend
        self.orig_forward = model.forward
        self.calls = 0
        self.forwards = 0
        self.patched_levels = set()

    def attend(self, Q, V):
        out = self.orig_attend(Q, V)
        level = self.calls
        self.calls += 1
        assert level < self.n_layer, f"_attend called {self.calls} times in one forward"
        new = self.transform(level, out)
        if new is not out:
            self.patched_levels.add(level)
        return new

    def forward(self, *args, **kwargs):
        self.calls = 0
        result = self.orig_forward(*args, **kwargs)
        assert self.calls == self.n_layer, (
            f"_attend was called {self.calls} times in one forward, expected {self.n_layer}"
        )
        self.forwards += 1
        return result


@contextmanager
def intervened(model, condition: str):
    """Install the interceptor for the duration. core.py is never touched."""
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
    expected = {"intact": [], "patched": list(range(N_LAYER)), "zeroed": list(range(N_LAYER))}.get(
        condition, [int(condition[len("patched_L") :])] if condition.startswith("patched_L") else None
    )
    assert patched_levels == expected, (
        f"{condition}: transformed levels {patched_levels}, expected {expected}"
    )
    total_n = sum(counts.values())
    total_ce = sum(sums.values())
    return counts, sums, total_n, total_ce, n_forwards, patched_levels


def summarize(counts, sums, total_n, total_ce, patched_levels):
    out = {
        "n_tokens": int(total_n),
        "mean_ce": total_ce / total_n,
        "patched_levels": patched_levels,
        "saturated": bool(total_ce / total_n > SATURATION_CE),
        "classes": {},
    }
    for c, name in CLASS_NAMES.items():
        n = counts[c]
        out["classes"][str(c)] = {
            "name": name,
            "n_tokens": int(n),
            "frac_of_tokens": n / total_n,
            "mean_ce": (sums[c] / n) if n else float("nan"),
            "share_of_loss": (sums[c] / total_ce) if total_ce else float("nan"),
        }
    return out


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


def headline(results, seeds):
    per_cond = {c: condition_stats(results, seeds, c) for c in CONDITIONS if c != "intact"}

    usable = [c for c in LEVEL_CONDITIONS if not per_cond[c]["mean"]["any_seed_saturated"]]
    pool = usable or LEVEL_CONDITIONS
    best = max(pool, key=lambda c: per_cond[c]["mean"]["excess"])
    best_level = int(best[len("patched_L") :])

    per_seed_best = {}
    for s in seeds:
        cand = [
            c
            for c in LEVEL_CONDITIONS
            if not results[f"seed{s}"][c]["saturated"]
        ] or LEVEL_CONDITIONS
        bl = max(cand, key=lambda c: deltas(results, s, c)[2] - deltas(results, s, c)[3])
        per_seed_best[str(s)] = int(bl[len("patched_L") :])

    return {
        "saturation_ce_threshold": SATURATION_CE,
        "saturated_conditions": [c for c in per_cond if per_cond[c]["mean"]["any_seed_saturated"]],
        "per_condition": per_cond,
        "most_selective_level_mean_over_seeds": best_level,
        "most_selective_level_per_seed": per_seed_best,
        "study2_probe_levels_per_seed": {str(k): v for k, v in STUDY2_LEVELS.items()},
        "agreement_per_seed": {
            str(s): bool(per_seed_best[str(s)] in STUDY2_LEVELS[s]) for s in seeds
        },
        "agreement_pooled": bool(
            best_level in sorted({lv for s in seeds for lv in STUDY2_LEVELS[s]})
        ),
    }


# ---------------------------------------------------------------- printing


def print_summary(results, seeds, h):
    print("=" * 104)
    print(f"PER-LEVEL yKV PATCHING, mean over seeds {seeds}   (EXPLORATORY, not preregistered)")
    print("=" * 104)
    print(
        f"{'condition':<12s} {'overallCE':>9s} {'d c1':>8s} {'d c3':>9s} {'d c2':>9s} "
        f"{'excess':>9s} {'ratio':>7s}  note"
    )
    intact_ce = float(np.mean([results[f"seed{s}"]['intact']['mean_ce'] for s in seeds]))
    print(
        f"{'intact':<12s} {intact_ce:9.4f} {0.0:8.4f} {0.0:9.4f} {0.0:9.4f} {0.0:9.4f} "
        f"{'--':>7s}  reference"
    )
    for cond in LEVEL_CONDITIONS + ["patched", "zeroed"]:
        m = h["per_condition"][cond]["mean"]
        note = "SATURATED - discard contrast" if m["any_seed_saturated"] else ""
        if cond == "patched":
            note = note or "all levels"
            note = "all levels; " + note if "SAT" in note else note
        if cond == "zeroed":
            note = ("all levels; " + note) if note else "all levels"
        print(
            f"{cond:<12s} {m['overall_ce']:9.4f} {m['d_class1']:8.4f} {m['d_class3']:9.4f} "
            f"{m['d_class2']:9.4f} {m['excess']:+9.4f} {m['ratio_of_means']:7.2f}  {note}"
        )
    print("\n  c1 = empty/wall, c2 = memory-relevant object, c3 = other object (control).")
    print("  excess = d(class2) - d(class3), in nats. ratio = mean d(c2) / mean d(c3).")

    print("\n" + "-" * 104)
    print("PER-SEED excess (nats) by patched level")
    print("-" * 104)
    hdr = f"{'seed':<6s}" + "".join(f"{c[-2:]:>12s}" for c in LEVEL_CONDITIONS) + f"{'argmax':>10s}"
    print(hdr)
    for s in seeds:
        row = f"{s:<6d}"
        for c in LEVEL_CONDITIONS:
            d = deltas(results, s, c)
            flag = "*" if results[f"seed{s}"][c]["saturated"] else " "
            row += f"{d[2] - d[3]:>+11.4f}{flag}"
        row += f"{'L' + str(h['most_selective_level_per_seed'][str(s)]):>10s}"
        print(row)
    print("  * = that seed/level is in the saturation regime (overall CE > "
          f"{SATURATION_CE}); its contrast is unreliable.")

    print("\n" + "-" * 104)
    print("CONVERGENCE with Study 2's sigma-probe level selection")
    print("-" * 104)
    for s in seeds:
        causal = h["most_selective_level_per_seed"][str(s)]
        probe = STUDY2_LEVELS[s]
        ok = "AGREE" if h["agreement_per_seed"][str(s)] else "DIVERGE"
        print(f"  seed {s}: most selective patch L{causal}   probe best-on-val {probe}   -> {ok}")
    print(
        f"  pooled: most selective L{h['most_selective_level_mean_over_seeds']} vs probe levels "
        f"{sorted({lv for s in seeds for lv in STUDY2_LEVELS[s]})} -> "
        f"{'AGREE' if h['agreement_pooled'] else 'DIVERGE'}"
    )
    print("-" * 104)
    print("Post-hoc and exploratory. Decides nothing; cannot revise H5-H8.")
    print("The all-levels `patched`/`zeroed` arms saturate by construction and are NOT a null.")


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true", help="seed 0, 100 episodes, no JSON written")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else ([0] if args.validate else SEEDS)
    n_ep = args.episodes if args.episodes is not None else (100 if args.validate else N_EPISODES)
    conditions = args.conditions or CONDITIONS
    assert "intact" in conditions, "intact is the baseline for every delta"

    device = select_device(None)
    print(f"EXPLORATORY post-hoc yKV causal patch | device={device} root={ROOT}")
    print(f"seeds={seeds} episodes={n_ep} chunk={args.chunk}")
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
        model, cfg, meta = load_checkpoint(str(ckpt), device)
        model.eval()
        assert model.hcfg.n_layer == N_LAYER, f"n_layer {model.hcfg.n_layer} != {N_LAYER}"
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
                f"  {cond:<12s} overall {s['mean_ce']:8.4f}{flag} | c1 {c1['mean_ce']:7.4f} | "
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

    h = headline(results, seeds)
    print_summary(results, seeds, h)

    if not args.validate:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": "EXPLORATORY, post-hoc, NOT preregistered. Cannot revise H5-H8.",
            "saturation_note": (
                "The all-levels `patched` and `zeroed` conditions are catastrophic, not selective: "
                "they take overall CE from ~0.024 to several nats and lift even empty/wall cells "
                "far above chance, so both object classes are pinned near a floor and the "
                "class-2 vs class-3 contrast carries no information. Their ~0.94x ratio is a "
                "ceiling artefact and must NOT be read as a null result for the associative read. "
                "They are retained only as the saturation reference; the per-level conditions are "
                "the informative arms."
            ),
            "run": RUN,
            "split": "probe_test",
            "n_episodes": int(n_ep),
            "chunk": int(args.chunk),
            "n_layer": N_LAYER,
            "conditions": conditions,
            "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
            "per_seed": results,
            "headline": h,
        }
        OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
