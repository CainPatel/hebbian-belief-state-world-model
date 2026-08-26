import json
import subprocess

import pytest

from hbwm import matrix
from hbwm.matrix import (
    GAMMA_ARMS,
    LRS,
    MODELS,
    best_lr,
    e1_jobs,
    e2_jobs,
    e3_jobs,
    headline_runs,
    is_done,
    run_name,
    run_path,
    train_cmd,
)


def _fake_done(root, exp, stem, lr, seed, val):
    p = run_path(root, exp, stem, lr, seed)
    p.mkdir(parents=True)
    (p / "final.json").write_text(json.dumps({"best_val_ce": val}))


def test_names_and_jobs(tmp_path):
    assert run_name("bdh_g100", 1e-3) == "bdh_g100_lr0.001"
    assert run_name("lstm", 3e-4) == "lstm_lr0.0003"
    assert len(e1_jobs()) == 9 and all(j[2] == 0 for j in e1_jobs())
    assert not is_done(run_path(tmp_path, "x", "lstm", 1e-3, 0))


def test_best_lr_and_followups(tmp_path):
    vals = {3e-4: 0.9, 1e-3: 0.5, 3e-3: 0.7}
    for stem in MODELS:
        for lr, v in vals.items():
            _fake_done(tmp_path, "x", stem, lr, 0, v + (0.1 if stem == "rwkv" and lr == 1e-3 else 0))
    assert best_lr(tmp_path, "x", "bdh_g100") == 1e-3
    assert best_lr(tmp_path, "x", "rwkv") == 1e-3  # 0.6 still the min
    e2 = e2_jobs(tmp_path, "x")
    assert len(e2) == 6 and {j[2] for j in e2} == {1, 2} and all(j[1] == 1e-3 for j in e2)
    e3 = e3_jobs(tmp_path, "x")
    assert len(e3) == 6 and {j[0] for j in e3} == set(GAMMA_ARMS) and all(j[1] == 1e-3 for j in e3)
    hl = headline_runs(tmp_path, "x")
    assert len(hl) == 15
    cmd = train_cmd(("lstm", 1e-3, 2), tmp_path)
    assert "--config" in cmd and "experiments/train/lstm.json" in cmd and "--seed" in cmd and "2" in cmd
    assert "--data-dir" not in cmd  # no data dir given -> train.py falls back to its config
    cmd = train_cmd(("lstm", 1e-3, 2), tmp_path, data="data/grid9")
    assert cmd[cmd.index("--data-dir") + 1] == "data/grid9"


def test_probes_phase_continues_past_a_failing_checkpoint(tmp_path, monkeypatch, capsys):
    """A probe run killed by the OOM killer must not discard the other 14 checkpoints of the phase."""
    for stem in MODELS:
        for lr in LRS:
            _fake_done(tmp_path, "x", stem, lr, 0, 0.5 if lr == 1e-3 else 0.9)
    attempted = []

    def fake_run(cmd, dry):
        attempted.append(cmd[cmd.index("--run-dir") + 1])
        if len(attempted) == 1:
            raise subprocess.CalledProcessError(-9, cmd)  # SIGKILL, as Jetsam delivers it

    monkeypatch.setattr(matrix, "_run", fake_run)
    monkeypatch.setattr("sys.argv", ["matrix", "--phase", "probes", "--root", str(tmp_path), "--exp", "x"])
    with pytest.raises(SystemExit) as exc:
        matrix.main()
    assert exc.value.code == 1  # the shell chain still stops before evaluate
    assert len(attempted) == len(headline_runs(tmp_path, "x")) == 15  # every later checkpoint was tried
    assert f"probes phase: 1 failed: {attempted[0]}" in capsys.readouterr().out


def test_probes_phase_exits_cleanly_when_every_checkpoint_succeeds(tmp_path, monkeypatch):
    for stem in MODELS:
        for lr in LRS:
            _fake_done(tmp_path, "x", stem, lr, 0, 0.5 if lr == 1e-3 else 0.9)
    attempted = []
    monkeypatch.setattr(matrix, "_run", lambda cmd, dry: attempted.append(cmd))
    monkeypatch.setattr("sys.argv", ["matrix", "--phase", "probes", "--root", str(tmp_path), "--exp", "x"])
    matrix.main()  # no SystemExit
    assert len(attempted) == 15
