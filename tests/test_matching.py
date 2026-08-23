import json
from pathlib import Path

from hbwm.baselines.lstm import LSTMConfig
from hbwm.baselines.matching import rel_err, solve_lstm_hidden, solve_rwkv_width
from hbwm.baselines.rwkv import RWKVConfig
from hbwm.models import build_model, count_params

TARGET = 1_577_216  # bdh_g100 reference


def test_solvers_hit_within_5pct():
    H = solve_lstm_hidden(TARGET, LSTMConfig())
    C = solve_rwkv_width(TARGET, RWKVConfig())
    assert H == 350 and C == 176
    from hbwm.baselines.lstm import LSTMLM
    from hbwm.baselines.rwkv import RWKVLM

    assert rel_err(count_params(LSTMLM(LSTMConfig(hidden=H))), TARGET) <= 0.05
    assert rel_err(count_params(RWKVLM(RWKVConfig(n_embd=C))), TARGET) <= 0.05


def test_preregistered_configs_are_matched():
    root = Path(__file__).resolve().parents[1] / "experiments" / "train"
    counts = {}
    for stem in ["bdh_g100", "lstm", "rwkv"]:
        cfg = json.loads((root / f"{stem}.json").read_text())
        counts[stem] = count_params(build_model(cfg["model"], cfg["model_cfg"]))
    assert counts["bdh_g100"] == TARGET
    assert rel_err(counts["lstm"], TARGET) <= 0.05 and rel_err(counts["rwkv"], TARGET) <= 0.05
