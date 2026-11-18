"""Small utilities: logging, atomic file writes, progress reporting, formatting."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("badapple_dr")


def setup_logging(work_dir: str | os.PathLike | None = None, verbose: bool = False) -> None:
    """Log to the console and (if work_dir is given) to work_dir/pipeline.log."""
    try:  # never crash on a Windows console that cannot print a character
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    LOG.setLevel(logging.DEBUG)
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
        h.close()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    LOG.addHandler(ch)
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(work_dir) / "pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        LOG.addHandler(fh)
    LOG.propagate = False


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean_nan(o):
    """Replace NaN/inf floats by None so the JSON stays standard."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    return o


def atomic_write_json(path: str | os.PathLike, obj) -> None:
    """Write JSON via a temporary file + os.replace so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean_nan(obj), f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str | os.PathLike, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_npz(path: str | os.PathLike, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fingerprint(obj) -> str:
    """Stable short hash of a JSON-serialisable object (used to detect config changes)."""
    s = json.dumps(_clean_nan(obj), sort_keys=True, default=_json_default)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def file_identity(path: str | os.PathLike) -> dict:
    p = Path(path)
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def fmt_timecode(seconds: float) -> str:
    """Seconds -> HH:MM:SS.mmm"""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return ""
    ms_total = int(round(seconds * 1000.0))
    sign = "-" if ms_total < 0 else ""
    ms_total = abs(ms_total)
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


class Progress:
    """Progress reporter.

    Uses a tqdm bar on an interactive console; otherwise (GUI, redirected output, logs) prints a
    status line with rate and ETA every `log_every` seconds.
    """

    def __init__(self, total=None, desc="", unit="it", initial=0, log_every=15.0):
        self.total = total
        self.n = initial
        self.n0 = initial
        self.desc = desc
        self.unit = unit
        self.t0 = time.time()
        self.last_log = 0.0
        self.last_logged_n = None
        self.log_every = log_every
        self.postfix: dict = {}
        self._bar = None
        if sys.stdout.isatty() and not os.environ.get("BADAPPLE_NO_TQDM"):
            try:
                from tqdm import tqdm

                self._bar = tqdm(total=total, initial=initial, desc=desc, unit=unit,
                                 dynamic_ncols=True, smoothing=0.05, unit_scale=True)
            except Exception:
                self._bar = None

    def update(self, k=1, **postfix):
        self.n += k
        if postfix:
            self.postfix.update(postfix)
        if self._bar is not None:
            self._bar.update(k)
            if postfix:
                self._bar.set_postfix(self.postfix, refresh=False)
        else:
            now = time.time()
            if now - self.last_log >= self.log_every:
                self.last_log = now
                self.last_logged_n = self.n
                LOG.info(self.status_line())

    def status_line(self) -> str:
        el = time.time() - self.t0
        done = self.n - self.n0
        rate = done / el if el > 0 else 0.0
        parts = [f"{self.desc}: {self.n:,}"]
        if self.total:
            pct = 100.0 * self.n / self.total if self.total else 0
            parts[0] += f" / {int(self.total):,} {self.unit} ({pct:.1f}%)"
            if rate > 0:
                parts.append(f"ETA {fmt_duration(max(0.0, (self.total - self.n) / rate))}")
        else:
            parts[0] += f" {self.unit}"
        parts.append(f"{rate:,.0f} {self.unit}/s")
        parts.append(f"elapsed {fmt_duration(el)}")
        if self.postfix:
            parts.append(", ".join(f"{k}={v}" for k, v in self.postfix.items()))
        return " | ".join(parts)

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None
            LOG.debug(self.status_line())
        elif self.last_logged_n != self.n:
            LOG.info(self.status_line())


_STOP_FILE: Path | None = None


def set_stop_file(work_dir: str | os.PathLike) -> None:
    """A file named STOP in the work folder asks the pipeline to save its progress and exit."""
    global _STOP_FILE
    _STOP_FILE = Path(work_dir) / "STOP"
    try:
        _STOP_FILE.unlink()
    except FileNotFoundError:
        pass


def check_stop() -> None:
    if _STOP_FILE is not None and _STOP_FILE.exists():
        raise KeyboardInterrupt("stop requested (STOP file)")


class StageTimer:
    """Accumulates wall-clock time per stage across resumed runs (stored in work/timings.json)."""

    def __init__(self, work_dir: str | os.PathLike):
        self.path = Path(work_dir) / "timings.json"
        self.data = read_json(self.path, {}) or {}

    def add(self, stage: str, seconds: float) -> None:
        self.data[stage] = float(self.data.get(stage, 0.0)) + float(seconds)
        atomic_write_json(self.path, self.data)

    def total(self) -> float:
        return float(sum(self.data.values()))
