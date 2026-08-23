import numpy as np

from hbwm.probes.decisions import h1_decision, h2_curve, h3_latency, h4_k90


def test_h1():
    r = h1_decision([0.8, 0.82, 0.79], {"lstm": [0.6, 0.7, 0.65], "x_sparse": [0.76, 0.7, 0.72]})
    assert r["comparators"]["lstm"]["passes"] and r["comparators"]["x_sparse"]["passes"] and r["supported"]
    r = h1_decision([0.8, 0.82, 0.79], {"lstm": [0.6, 0.83, 0.65]})  # one paired diff negative
    assert not r["supported"]
    r = h1_decision([0.8, 0.82, 0.79], {"lstm": [0.77, 0.78, 0.76]})  # mean margin < 5 pts
    assert not r["supported"] and abs(r["comparators"]["lstm"]["mean_diff"] - 0.0333) < 1e-3


def test_h2():
    ok = {"1-4": 0.9, "5-8": 0.85, "9-16": 0.8, "17-32": 0.7, "33-64": 0.5, "65+": 0.4}
    assert h2_curve(ok)["graceful"]
    cliff = dict(ok, **{"9-16": 0.3})  # drops to < 50% of predecessor
    assert not h2_curve(cliff)["graceful"]
    low = dict(ok, **{"33-64": 0.4})
    assert not h2_curve(low)["graceful"]
    assert h2_curve(dict(ok, **{"65+": None}))["graceful"]


def test_h3():
    ep = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    steps = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    p_old = np.array([0.9, 0.6, 0.2, 0.9, 0.9, 0.9, 0.1, 0.1])
    p_new = np.array([0.1, 0.4, 0.8, 0.1, 0.1, 0.1, 0.9, 0.9])
    r = h3_latency(p_old, p_new, steps, ep)
    assert r["n_episodes"] == 3 and r["n_flipped"] == 2 and r["latencies"] == [2, 0]
    assert abs(r["frac_le5"] - 2 / 3) < 1e-9 and not r["supported"]


def test_h3_no_flip():
    ep = np.array([0, 0, 1, 1])
    steps = np.array([0, 1, 0, 1])
    p_old = np.array([0.9, 0.9, 0.9, 0.9])
    p_new = np.array([0.1, 0.1, 0.1, 0.1])
    r = h3_latency(p_old, p_new, steps, ep)
    assert r["n_episodes"] == 2 and r["n_flipped"] == 0 and r["latencies"] == []
    assert r["median_latency"] is None and r["frac_le5"] == 0.0 and r["supported"] is False


def test_h4():
    r = h4_k90({16: 0.3, 64: 0.5, 256: 0.74, 1024: 0.8}, acc_all=0.8, n_features=524288)
    assert r["k90"] == 256 and r["strong"] and r["weak"]
    r = h4_k90({16: 0.3, 64: 0.5}, acc_all=0.8, n_features=1000)
    assert r["k90"] is None and not r["strong"] and not r["weak"]
    r = h4_k90({1024: 0.75}, acc_all=0.8, n_features=50000)
    assert r["k90"] == 1024 and not r["strong"] and not r["weak"]  # 1024 > 1% of 50000
