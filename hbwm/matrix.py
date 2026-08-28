"""Study 1 experiment matrix (spec section 5): E1 LR sweep, E2 seeds, E3 gamma arms, probes, evaluate."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

MODELS = ["bdh_g100", "lstm", "rwkv"]
GAMMA_ARMS = ["bdh_g099", "bdh_g097"]
LRS = [3e-4, 1e-3, 3e-3]
SEEDS = [0, 1, 2]


def run_name(stem: str, lr: float) -> str:
    return f"{stem}_lr{lr:g}"


def run_path(root, exp: str, stem: str, lr: float, seed: int) -> Path:
    return Path(root) / exp / run_name(stem, lr) / f"seed{seed}"


def is_done(path) -> bool:
    return (Path(path) / "final.json").exists()


def e1_jobs():
    return [(stem, lr, 0) for stem in MODELS for lr in LRS]


def best_lr(root, exp, stem) -> float:
    vals = {}
    for lr in LRS:
        f = run_path(root, exp, stem, lr, 0) / "final.json"
        if not f.exists():
            raise FileNotFoundError(f"E1 run missing: {f}")
        v = json.loads(f.read_text())["best_val_ce"]
        vals[lr] = v if (v is not None and math.isfinite(v)) else float("inf")  # None/NaN/inf = diverged = worst
    return min(vals, key=vals.get)


def e2_jobs(root, exp):
    return [(stem, best_lr(root, exp, stem), seed) for stem in MODELS for seed in SEEDS if seed != 0]


def e3_jobs(root, exp):
    lr = best_lr(root, exp, "bdh_g100")
    return [(stem, lr, seed) for stem in GAMMA_ARMS for seed in SEEDS]


def headline_runs(root, exp):
    jobs = [(stem, best_lr(root, exp, stem), seed) for stem in MODELS for seed in SEEDS]
    return jobs + e3_jobs(root, exp)


STUDY2_LEVELS = {0: [3, 4], 1: [3], 2: [4]}  # spec section 6: each seed's Study 1 best levels
STUDY2_LR = 0.003  # RESULTS.md: best_lr.json selected 3e-3 for all three families; the single
# source of truth for the LR-tagged run directory both this module and hbwm.probes.evaluate read


def study2_jobs():
    return [(stem, STUDY2_LR, seed) for stem in MODELS for seed in SEEDS]


def train_cmd(job, root, data=None):
    stem, lr, seed = job
    cmd = [sys.executable, "-m", "hbwm.train", "--config", f"experiments/train/{stem}.json",
           "--seed", str(seed), "--lr", str(lr), "--out-root", str(root)]
    return cmd + (["--data-dir", str(data)] if data is not None else [])


def _run(cmd, dry):
    print(" ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                     choices=["e1", "e2", "e3", "probes", "evaluate", "study2", "study2-evaluate"])
    ap.add_argument("--root", default="runs")
    ap.add_argument("--exp", default="study1")
    ap.add_argument("--data", default="data/grid9")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root, exp = args.root, args.exp
    if args.phase in ("e1", "e2", "e3"):
        jobs = {"e1": lambda: e1_jobs(), "e2": lambda: e2_jobs(root, exp), "e3": lambda: e3_jobs(root, exp)}[args.phase]()
        for job in jobs:
            if is_done(run_path(root, exp, *job)):
                print(f"skip (done): {job}")
                continue
            _run(train_cmd(job, root, args.data), args.dry_run)
        if args.phase != "e1":
            (Path(root) / exp / "best_lr.json").write_text(
                json.dumps({stem: best_lr(root, exp, stem) for stem in MODELS}, indent=2) + "\n")
    elif args.phase == "probes":
        failed = []
        for job in headline_runs(root, exp):
            rd = run_path(root, exp, *job)
            if (rd / "probes" / "done.json").exists():
                print(f"skip (probed): {rd}")
                continue
            try:
                _run([sys.executable, "-m", "hbwm.probes.run", "--run-dir", str(rd), "--data", args.data,
                      "--preset", "study1"], args.dry_run)
            except subprocess.CalledProcessError as e:  # e.g. an OOM kill: rerunning re-probes only this one
                # A SIGKILL is uncatchable, so run_probes' own `finally` never ran and its fp16
                # sigma_full memmaps (~25 GB per level) are stranded. This process is the one left
                # standing, and the next checkpoint needs that disk space.
                shutil.rmtree(rd / "probes" / "cache", ignore_errors=True)
                print(f"probe run failed (rc={e.returncode}): {rd}", flush=True)
                failed.append(str(rd))
        print(f"probes phase: {len(failed)} failed: {' '.join(failed)}", flush=True)
        if failed:
            sys.exit(1)  # the shell chain must still stop before evaluate
    elif args.phase == "study2":
        failed = []
        for stem, lr, seed in study2_jobs():
            rd = run_path(root, args.exp, stem, lr, seed)
            if (rd / "probes2" / "done.json").exists():
                print(f"skip (probed): {rd}")
                continue
            cmd = [sys.executable, "-m", "hbwm.probes.run", "--run-dir", str(rd), "--data", args.data,
                   "--study", "2"]
            if stem.startswith("bdh"):
                cmd += ["--levels", ",".join(str(x) for x in STUDY2_LEVELS[seed])]
            try:
                _run(cmd, args.dry_run)
            except subprocess.CalledProcessError as e:
                shutil.rmtree(rd / "probes2" / "cache", ignore_errors=True)
                print(f"study2 probe run failed (rc={e.returncode}): {rd}", flush=True)
                failed.append(str(rd))
        print(f"study2 phase: {len(failed)} failed: {' '.join(failed)}", flush=True)
        if failed:
            sys.exit(1)
    elif args.phase == "study2-evaluate":
        _run([sys.executable, "-m", "hbwm.probes.evaluate", "--root", root, "--exp", args.exp,
              "--data", args.data, "--study", "2"], args.dry_run)
    else:
        _run([sys.executable, "-m", "hbwm.probes.evaluate", "--root", root, "--exp", exp,
              "--data", args.data], args.dry_run)


if __name__ == "__main__":
    main()
