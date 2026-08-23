import argparse
import dataclasses
import json

from hbwm.baselines.lstm import LSTMLM, LSTMConfig
from hbwm.baselines.rwkv import RWKVLM, RWKVConfig
from hbwm.models import build_model, count_params


def rel_err(n: int, target: int) -> float:
    return abs(n - target) / target


def _closest(counter, target, lo, hi, step=1):
    """counter is monotone increasing on the integer grid lo..hi (step). Binary search, return closest."""
    cand = list(range(lo, hi + 1, step))
    a, b = 0, len(cand) - 1
    while b - a > 1:
        mid = (a + b) // 2
        if counter(cand[mid]) < target:
            a = mid
        else:
            b = mid
    return min((cand[a], cand[b]), key=lambda x: abs(counter(x) - target))


def solve_lstm_hidden(target: int, base: LSTMConfig) -> int:
    return _closest(lambda H: count_params(LSTMLM(dataclasses.replace(base, hidden=H))), target, 8, 1024)


def solve_rwkv_width(target: int, base: RWKVConfig, step: int = 8) -> int:
    return _closest(lambda C: count_params(RWKVLM(dataclasses.replace(base, n_embd=C))), target, 16, 1024, step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdh-config", default="experiments/train/bdh_g100.json")
    args = ap.parse_args()
    cfg = json.load(open(args.bdh_config))
    target = count_params(build_model(cfg["model"], cfg["model_cfg"]))
    H = solve_lstm_hidden(target, LSTMConfig())
    C = solve_rwkv_width(target, RWKVConfig())
    nl = count_params(LSTMLM(LSTMConfig(hidden=H)))
    nr = count_params(RWKVLM(RWKVConfig(n_embd=C)))
    print(json.dumps({"target": target, "lstm_hidden": H, "lstm_params": nl, "lstm_rel_err": rel_err(nl, target),
                      "rwkv_width": C, "rwkv_params": nr, "rwkv_rel_err": rel_err(nr, target)}, indent=2))


if __name__ == "__main__":
    main()
