import dataclasses
import json

import pytest

from hbwm.config import from_dict, load_config, save_config, to_dict


@dataclasses.dataclass
class Inner:
    a: int = 1
    b: str = "x"


@dataclasses.dataclass
class Outer:
    inner: Inner = dataclasses.field(default_factory=Inner)
    n: float = 2.0
    tags: dict = dataclasses.field(default_factory=dict)


def test_nested_roundtrip(tmp_path):
    o = Outer(inner=Inner(a=5, b="y"), n=3.5, tags={"k": 1})
    p = tmp_path / "c.json"
    save_config(o, p)
    assert json.loads(p.read_text())["inner"]["a"] == 5
    assert load_config(p, Outer) == o


def test_partial_dict_uses_defaults():
    o = from_dict(Outer, {"inner": {"a": 9}})
    assert o.inner == Inner(a=9, b="x") and o.n == 2.0


def test_unknown_field_raises():
    with pytest.raises(ValueError):
        from_dict(Outer, {"bogus": 1})


def test_to_dict():
    assert to_dict(Inner()) == {"a": 1, "b": "x"}
