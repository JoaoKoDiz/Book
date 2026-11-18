"""Append-only on-disk cache of analysis images keyed by source frame index.

Used for frames re-decoded at the fine level (stage 5) and for frames found by dense refinement
(stage 7), so that an interrupted run never has to fetch them again.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .util import atomic_save_npz

IDX_DTYPE = np.dtype([("n", "<i8"), ("pts", "<f8"), ("row", "<i8")])


class FrameStore:
    def __init__(self, directory: str | os.PathLike, shape: tuple[int, ...]):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(int(s) for s in shape)
        self.fb = int(np.prod(self.shape))
        self.data_path = self.dir / "frames.u8"
        self.index_path = self.dir / "index.npz"
        self.map: dict[int, tuple[int, float]] = {}
        rows = 0
        if self.index_path.exists():
            d = np.load(self.index_path)
            if tuple(d["shape"].tolist()) != self.shape:
                raise RuntimeError(f"frame cache {self.dir} has shape {d['shape']} != {self.shape}")
            idx = d["idx"]
            size = self.data_path.stat().st_size if self.data_path.exists() else 0
            valid = idx["row"] < size // self.fb
            idx = idx[valid]
            for n, pts, row in idx:
                self.map[int(n)] = (int(row), float(pts))
            rows = int(idx["row"].max()) + 1 if len(idx) else 0
        if self.data_path.exists():
            with open(self.data_path, "r+b") as f:
                f.truncate(rows * self.fb)
        else:
            self.data_path.touch()
        self.rows = rows

    def __contains__(self, n) -> bool:
        return int(n) in self.map

    def __len__(self) -> int:
        return len(self.map)

    def pts(self, n) -> float:
        return self.map[int(n)][1]

    def add_many(self, items) -> None:
        """items: iterable of (n, pts, array)."""
        new = [(int(n), float(p), a) for n, p, a in items if int(n) not in self.map]
        if not new:
            return
        with open(self.data_path, "ab") as f:
            for n, p, a in new:
                a = np.ascontiguousarray(a, np.uint8)
                if a.shape != self.shape:
                    raise ValueError(f"bad frame shape {a.shape} (expected {self.shape})")
                f.write(a.tobytes())
                self.map[n] = (self.rows, p)
                self.rows += 1
            f.flush()
            os.fsync(f.fileno())
        self._save_index()

    def _save_index(self) -> None:
        idx = np.array([(n, p, r) for n, (r, p) in self.map.items()], IDX_DTYPE)
        atomic_save_npz(self.index_path, idx=idx, shape=np.array(self.shape))

    def get_many(self, ns) -> np.ndarray:
        ns = [int(n) for n in ns]
        out = np.zeros((len(ns),) + self.shape, np.uint8)
        order = sorted(range(len(ns)), key=lambda i: self.map[ns[i]][0])
        with open(self.data_path, "rb") as f:
            for i in order:
                row = self.map[ns[i]][0]
                f.seek(row * self.fb)
                out[i] = np.frombuffer(f.read(self.fb), np.uint8).reshape(self.shape)
        return out
