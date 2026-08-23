import dataclasses
import json
import typing
from pathlib import Path


def from_dict(cls, d: dict):
    """Build dataclass `cls` from a (possibly nested) dict. Unknown keys raise."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"unknown fields for {cls.__name__}: {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in d.items():
        t = hints.get(name)
        if dataclasses.is_dataclass(t) and isinstance(value, dict):
            value = from_dict(t, value)
        kwargs[name] = value
    return cls(**kwargs)


def to_dict(obj) -> dict:
    return dataclasses.asdict(obj)


def load_config(path, cls):
    return from_dict(cls, json.loads(Path(path).read_text()))


def save_config(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(obj), indent=2) + "\n")
