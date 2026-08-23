import dataclasses

import numpy as np

BUCKET_EDGES = [1, 5, 9, 17, 33, 65]
BUCKET_NAMES = ["1-4", "5-8", "9-16", "17-32", "33-64", "65+"]


def bucket_of(sss) -> np.ndarray:
    return np.digitize(np.asarray(sss), BUCKET_EDGES) - 1


def cell_id(xy, G: int) -> np.ndarray:
    xy = np.asarray(xy)
    return xy[..., 1] * G + xy[..., 0]


def eligible_mask(d) -> np.ndarray:
    return (d.steps_since_seen >= 1) & ~d.visible & ~d.stale


def last_seen_cell(d) -> np.ndarray:
    """Cell id of each object at its last visible step <= t (the oracle-memory prediction); -1 if never."""
    n, T, k = d.visible.shape
    t_idx = np.broadcast_to(np.arange(T)[None, :, None], (n, T, k))
    last_t = np.maximum.accumulate(np.where(d.visible, t_idx, -1), axis=1)
    cells = cell_id(d.obj_pos, d.G)
    out = np.take_along_axis(cells, np.clip(last_t, 0, None), axis=1)
    return np.where(last_t >= 0, out, -1)


@dataclasses.dataclass
class PairSet:
    ep: np.ndarray
    t: np.ndarray
    obj: np.ndarray
    label: np.ndarray
    bucket: np.ndarray
    oracle: np.ndarray
    sss: np.ndarray

    def __len__(self):
        return len(self.ep)

    def subset(self, idx):
        return PairSet(**{f.name: getattr(self, f.name)[idx] for f in dataclasses.fields(self)})

    def save(self, path):
        np.savez(path, **{f.name: getattr(self, f.name) for f in dataclasses.fields(self)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{k: z[k] for k in z.files})


def sample_pairs(d, rng: np.random.Generator, per_obj: int = 8) -> PairSet:
    """Up to per_obj eligible (t) per (episode, object), round-robin across steps_since_seen buckets."""
    el = eligible_mask(d)
    bk = bucket_of(d.steps_since_seen)
    oracle = last_seen_cell(d)
    cells = cell_id(d.obj_pos, d.G)
    E, T, OB, L_, B_, S_ = [], [], [], [], [], []
    for ep in range(d.n):
        for k in range(d.n_obj):
            ts = np.where(el[ep, :, k])[0]
            if len(ts) == 0:
                continue
            groups = {}
            for t in ts:
                groups.setdefault(int(bk[ep, t, k]), []).append(int(t))
            for g in groups.values():
                rng.shuffle(g)
            chosen = []
            keys = sorted(groups)
            while len(chosen) < per_obj and any(groups[b] for b in keys):
                for b in keys:
                    if groups[b] and len(chosen) < per_obj:
                        chosen.append(groups[b].pop())
            for t in chosen:
                E.append(ep)
                T.append(t)
                OB.append(k)
                L_.append(int(cells[ep, t, k]))
                B_.append(int(bk[ep, t, k]))
                S_.append(int(d.steps_since_seen[ep, t, k]))
    E, T, OB = np.array(E, dtype=np.int64), np.array(T, dtype=np.int64), np.array(OB, dtype=np.int64)
    return PairSet(ep=E, t=T, obj=OB, label=np.array(L_, dtype=np.int64), bucket=np.array(B_, dtype=np.int64),
                   oracle=oracle[E, T, OB] if len(E) else np.zeros(0, dtype=np.int64), sss=np.array(S_, dtype=np.int64))


@dataclasses.dataclass
class H3Pairs:
    ep: np.ndarray
    t: np.ndarray
    obj: np.ndarray
    old_cell: np.ndarray
    new_cell: np.ndarray
    steps_since_reobs: np.ndarray
    visible_now: np.ndarray

    def __len__(self):
        return len(self.ep)


def h3_pairs(d) -> H3Pairs:
    """All (t >= reobserved_t) for the moved object in moved-and-re-observed episodes."""
    E, T, OB, OLD, NEW, S, V = [], [], [], [], [], [], []
    for ep in np.where(d.moved & (d.reobserved_t >= 0))[0]:
        k, tr = int(d.move_obj[ep]), int(d.reobserved_t[ep])
        for t in range(tr, d.L + 1):
            E.append(ep)
            T.append(t)
            OB.append(k)
            OLD.append(int(cell_id(d.move_from[ep], d.G)))
            NEW.append(int(cell_id(d.move_to[ep], d.G)))
            S.append(t - tr)
            V.append(bool(d.visible[ep, t, k]))
    arr = lambda x, dt=np.int64: np.array(x, dtype=dt)  # noqa: E731
    return H3Pairs(ep=arr(E), t=arr(T), obj=arr(OB), old_cell=arr(OLD), new_cell=arr(NEW),
                   steps_since_reobs=arr(S), visible_now=arr(V, bool))
