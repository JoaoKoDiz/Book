"""Stage 2/3 -- build the searchable database of gameplay frames.

The whole source video is decoded ONCE, sequentially. ffmpeg crops the game area (analysis only),
scales it to a small grey thumbnail (64x48 for 4:3) and reports each frame's exact timestamp.
Python then decides which frames to keep (adaptive sampling):

* a frame is kept when it differs from the last kept frame by >= keep_threshold
  (static scenes -> sparse, movement / battles / cutscenes -> dense, automatically);
* frames around cuts, flashes and fast transitions are kept densely (cut_window frames each side);
* at least one frame every max_gap seconds is kept (coverage of long static scenes);
* identical-looking frames (same quantised thumbnail hash) are stored at most dup_limit times in
  the whole database (long black screens, menus, repeated dialogue boxes).

Storage (append-only, crash-safe, resumable):
    db/thumbs.u8   rows x H x W uint8 thumbnails
    db/meta.bin    rows x META_DTYPE  (frame index, exact timestamp, statistics, flags)
    db/state.json  committed rows / decoded frames / last timestamp / fingerprint
    db/density.npy per-minute counts of decoded and kept frames (sampling density report)
On restart, files are truncated to the last committed state and decoding resumes right after the
last committed frame (exact frame indices are preserved).
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from . import features as F
from .ffmpeg import FrameReader
from .util import LOG, Progress, atomic_write_json, fingerprint, read_json

META_DTYPE = np.dtype([
    ("n", "<i8"),        # frame index in the source video (0-based, presentation order)
    ("pts", "<f8"),      # exact raw timestamp (seconds) as stored in the file
    ("mean", "<f4"),     # mean brightness of the thumbnail (0..1)
    ("std", "<f4"),      # brightness standard deviation
    ("dprev", "<f4"),    # change vs previous decoded frame
    ("dkept", "<f4"),    # change vs previous kept frame
    ("flags", "<u1"),
])
FLAG_CUT, FLAG_NEAR_CUT, FLAG_GAP, FLAG_FIRST = 1, 2, 4, 8


class SourceDB:
    def __init__(self, db_dir: str | os.PathLike):
        self.dir = Path(db_dir)
        self.thumbs_path = self.dir / "thumbs.u8"
        self.meta_path = self.dir / "meta.bin"
        self.state_path = self.dir / "state.json"
        self.density_path = self.dir / "density.npy"
        self.prev_path = self.dir / "prev_pooled.npy"
        self._thumbs = None
        self._meta = None

    def state(self) -> dict | None:
        return read_json(self.state_path)

    def is_complete(self) -> bool:
        s = self.state()
        return bool(s and s.get("complete"))

    def open(self):
        s = self.state()
        if not s:
            raise RuntimeError(f"No source database in {self.dir} -- run the index stage first.")
        rows = int(s["committed_rows"])
        H, W = int(s["height"]), int(s["width"])
        self._thumbs = np.memmap(self.thumbs_path, np.uint8, "r", shape=(rows, H, W)) if rows else np.zeros((0, H, W), np.uint8)
        self._meta = np.fromfile(self.meta_path, META_DTYPE, count=rows)
        return self

    @property
    def thumbs(self) -> np.ndarray:
        if self._thumbs is None:
            self.open()
        return self._thumbs  # type: ignore[return-value]

    @property
    def meta(self) -> np.ndarray:
        if self._meta is None:
            self.open()
        return self._meta  # type: ignore[return-value]

    def close(self):
        if isinstance(self._thumbs, np.memmap):
            del self._thumbs
        self._thumbs = None
        self._meta = None

    def rows_before(self, pts: float) -> int:
        """Row of the last database frame with timestamp <= pts (or 0)."""
        j = int(np.searchsorted(self.meta["pts"], pts, side="right")) - 1
        return max(0, j)


def index_fingerprint(cfg: dict, src_info: dict, area: dict, mid: tuple[int, int]) -> str:
    keys = {k: cfg["index"][k] for k in ("keep_threshold", "max_gap", "cut_threshold", "cut_window", "dup_limit")}
    return fingerprint({"v": 1, "src": {"size": src_info["size_bytes"], "dur": src_info.get("duration"),
                                        "w": src_info["video"]["width"], "h": src_info["video"]["height"]},
                        "area": [area["x"], area["y"], area["w"], area["h"]], "mid": list(mid), "index": keys})


def _truncate(path: Path, nbytes: int) -> None:
    if path.exists():
        with open(path, "r+b") as f:
            f.truncate(nbytes)
    else:
        path.touch()


def build_index(ffmpeg: str, cfg: dict, src_info: dict, db_dir: str | os.PathLike, mid: tuple[int, int],
                stop_after_frames: int | None = None) -> dict:
    """Build (or resume building) the source database. Returns the final state dict."""
    icfg = cfg["index"]
    db = SourceDB(db_dir)
    db.dir.mkdir(parents=True, exist_ok=True)
    area = src_info["active_area"]
    W, H = mid
    fp = index_fingerprint(cfg, src_info, area, mid)
    v = src_info["video"]
    fps = float(v["fps_float"])
    duration = float(src_info.get("duration") or v.get("duration") or 0.0)
    file_start = float(src_info.get("start_time") or 0.0)
    total_est = int(round(duration * fps)) if duration else None
    frame_bytes = W * H

    state = db.state()
    if state and state.get("fingerprint") != fp:
        raise RuntimeError(
            f"The existing database in {db.dir} was built from a different source file or with different "
            "index settings. Use a new --work folder, or delete the 'db' folder inside it to rebuild.")
    if state and state.get("complete"):
        LOG.info("Source database already complete: %s frames decoded, %s kept.",
                 f"{state['decoded']:,}", f"{state['committed_rows']:,}")
        return state

    rows = int(state["committed_rows"]) if state else 0
    decoded = int(state["decoded"]) if state else 0
    last_pts = float(state["last_pts"]) if state and state.get("last_pts") is not None else None
    elapsed_before = float(state.get("elapsed", 0.0)) if state else 0.0
    _truncate(db.thumbs_path, rows * frame_bytes)
    _truncate(db.meta_path, rows * META_DTYPE.itemsize)

    n_min = int(duration // 60) + 2 if duration else 1
    density = np.zeros((n_min, 2), np.int64)
    if state and db.density_path.exists():
        d0 = np.load(db.density_path)
        density[: min(len(d0), n_min)] = d0[: min(len(d0), n_min)]

    # rebuild duplicate counters / last-kept / prev state from what is already stored
    dup_counts: dict = defaultdict(int)
    last_kept_pooled = None
    last_kept_pts = -1e18
    if rows:
        # read with real copies and release the memory map before the file is reopened for appending
        # (an open mapping can block file operations on Windows)
        th = np.memmap(db.thumbs_path, np.uint8, "r", shape=(rows, H, W))
        for i in range(0, rows, 65536):
            blk = np.array(th[i:i + 65536])
            for r in range(blk.shape[0]):
                dup_counts[F.quant_hash(blk[r])] += 1
        last_kept_pooled = F.pool2(F.to_float(np.array(th[rows - 1])))
        meta_last = np.fromfile(db.meta_path, META_DTYPE, count=rows)[-1]
        last_kept_pts = float(meta_last["pts"])
        th._mmap.close()  # type: ignore[attr-defined]
        del th
    prev_pooled = np.load(db.prev_path) if (state and db.prev_path.exists()) else None

    keep_thr = float(icfg["keep_threshold"])
    max_gap = float(icfg["max_gap"])
    cut_thr = float(icfg["cut_threshold"])
    cut_win = int(icfg["cut_window"])
    dup_limit = int(icfg["dup_limit"])

    seek = None
    if last_pts is not None:
        seek = last_pts - file_start - 1.0  # decode a little earlier; frames <= last_pts are skipped below
        LOG.info("Resuming source indexing after frame %s (t=%.3fs), %s frames already kept.",
                 f"{decoded - 1:,}", last_pts - file_start, f"{rows:,}")
    reader = FrameReader(ffmpeg, src_info["path"], pix_fmt="gray", size=(W, H),
                         crop=(area["x"], area["y"], area["w"], area["h"]),
                         seek=seek if (seek is not None and seek > 0) else None,
                         threads=int(icfg.get("threads") or 0), hwaccel=icfg.get("hwaccel"))

    f_thumbs = open(db.thumbs_path, "ab")
    f_meta = open(db.meta_path, "ab")
    pend_t: list[np.ndarray] = []
    pend_m: list[tuple] = []
    ring: deque = deque(maxlen=max(1, cut_win))
    force_keep = 0
    t_start = time.time()
    last_commit_t = time.time()
    last_commit_decoded = decoded
    complete = False
    prog = Progress(total=total_est, desc="Indexing source", unit="frames", initial=decoded)
    first_after_resume = last_pts is not None

    def commit(final: bool = False):
        nonlocal rows, last_commit_t, last_commit_decoded
        if pend_t:
            f_thumbs.write(np.stack(pend_t).tobytes())
            f_meta.write(np.array(pend_m, META_DTYPE).tobytes())
        f_thumbs.flush()
        f_meta.flush()
        os.fsync(f_thumbs.fileno())
        os.fsync(f_meta.fileno())
        rows += len(pend_t)
        pend_t.clear()
        pend_m.clear()
        if prev_pooled is not None:
            np.save(db.prev_path, prev_pooled)
        np.save(db.density_path, density)
        atomic_write_json(db.state_path, {
            "version": 1, "fingerprint": fp, "source": src_info["path"], "width": W, "height": H,
            "committed_rows": rows, "decoded": decoded, "last_pts": last_pts, "fps": fps,
            "file_start": file_start, "duration": duration, "complete": bool(final and complete),
            "elapsed": elapsed_before + (time.time() - t_start),
            "settings": {k: icfg[k] for k in ("keep_threshold", "max_gap", "cut_threshold", "cut_window", "dup_limit")},
        })
        last_commit_t = time.time()
        last_commit_decoded = decoded

    def keep_frame(n, pts, u8, pooled, dprev, dkept, flags, check_dup=True) -> bool:
        nonlocal last_kept_pooled, last_kept_pts
        h = F.quant_hash(u8)
        if check_dup and dup_counts[h] >= dup_limit:
            return False
        dup_counts[h] += 1
        x = pooled
        pend_t.append(u8.copy())
        pend_m.append((n, pts, float(x.mean()), float(x.std()), dprev, dkept, flags))
        last_kept_pooled = pooled
        last_kept_pts = pts
        density[min(int((pts - file_start) // 60), n_min - 1), 1] += 1
        return True

    try:
        while True:
            pts_b, fr_b = reader.read_batch(512)
            if len(pts_b) == 0:
                complete = stop_after_frames is None or decoded < stop_after_frames
                break
            if first_after_resume:
                sel = pts_b > last_pts + 1e-6
                if not sel.any():
                    continue
                k0 = int(np.argmax(sel))
                pts_b, fr_b = pts_b[k0:], fr_b[k0:]
                first_after_resume = False
            pooled_b = F.pool2(F.to_float(fr_b))
            for j in range(len(pts_b)):
                n = decoded
                pts = float(pts_b[j])
                u8 = fr_b[j]
                cur = pooled_b[j]
                dprev = float(np.abs(cur - prev_pooled).mean()) if prev_pooled is not None else 1.0
                dkept = float(np.abs(cur - last_kept_pooled).mean()) if last_kept_pooled is not None else 1.0
                flags = 0
                if last_kept_pooled is None:
                    keep, flags = True, FLAG_FIRST
                else:
                    keep = dkept >= keep_thr
                    if not keep and pts - last_kept_pts >= max_gap:
                        keep, flags = True, FLAG_GAP
                if dprev >= cut_thr and prev_pooled is not None:
                    flags |= FLAG_CUT
                    keep = True
                    force_keep = cut_win
                    for (rn, rpts, ru8, rpool, rdp) in list(ring):  # frames just before the cut
                        if rpts > last_kept_pts:
                            keep_frame(rn, rpts, ru8, rpool, rdp, 0.0, FLAG_NEAR_CUT)
                    ring.clear()
                elif force_keep > 0 and not keep:
                    keep = True
                    flags |= FLAG_NEAR_CUT
                    force_keep -= 1
                elif force_keep > 0:
                    force_keep -= 1
                kept = keep and keep_frame(n, pts, u8, cur, dprev, dkept, flags)
                if kept:
                    ring.clear()
                else:
                    ring.append((n, pts, u8.copy(), cur, dprev))
                density[min(int((pts - file_start) // 60), n_min - 1), 0] += 1
                prev_pooled = cur
                decoded += 1
                last_pts = pts
                if stop_after_frames is not None and decoded >= stop_after_frames:
                    break
            prog.update(len(pts_b), kept=f"{rows + len(pend_t):,}",
                        density=f"{100.0 * (rows + len(pend_t)) / max(decoded, 1):.1f}%")
            if (decoded - last_commit_decoded >= int(icfg["commit_frames"])
                    or time.time() - last_commit_t >= float(icfg["commit_seconds"])):
                commit()
            if stop_after_frames is not None and decoded >= stop_after_frames:
                LOG.info("Stopping early after %s frames (requested).", f"{decoded:,}")
                break
    except KeyboardInterrupt:
        LOG.warning("Interrupted -- saving progress (resume later with the same command).")
        commit(final=False)
        reader.close()
        f_thumbs.close()
        f_meta.close()
        raise
    finally:
        prog.close()
    reader.close()
    commit(final=True)
    f_thumbs.close()
    f_meta.close()
    st = db.state()
    if complete:
        if total_est and abs(decoded - total_est) > max(5, 0.002 * total_est):
            LOG.warning("Decoded %s frames, expected about %s from duration x fps.", f"{decoded:,}", f"{total_est:,}")
        LOG.info("Source database complete: %s frames decoded, %s kept (%.1f%%), %.1f MB.", f"{decoded:,}",
                 f"{rows:,}", 100.0 * rows / max(decoded, 1), rows * frame_bytes / 1e6)
    return st


def database_report(db: SourceDB, chapters: list | None = None) -> dict:
    """Sampling-density statistics for the quality report."""
    st = db.state() or {}
    meta = db.meta
    out = {
        "frames_decoded_and_analysed": int(st.get("decoded", 0)),
        "frames_in_database": int(len(meta)),
        "database_fraction": float(len(meta) / max(1, st.get("decoded", 1))),
        "kept_by_reason": {
            "change": int(((meta["flags"] & (FLAG_CUT | FLAG_NEAR_CUT | FLAG_GAP | FLAG_FIRST)) == 0).sum()),
            "cut_or_flash": int(((meta["flags"] & FLAG_CUT) != 0).sum()),
            "near_cut": int(((meta["flags"] & FLAG_NEAR_CUT) != 0).sum()),
            "max_gap": int(((meta["flags"] & FLAG_GAP) != 0).sum()),
        },
        "index_settings": st.get("settings"),
        "index_seconds": st.get("elapsed"),
    }
    if db.density_path.exists():
        d = np.load(db.density_path)
        dec, kept = d[:, 0], d[:, 1]
        valid = dec > 0
        if valid.any():
            frac = kept[valid] / dec[valid]
            out["kept_fraction_per_minute"] = {
                "min": float(frac.min()), "median": float(np.median(frac)), "max": float(frac.max()),
                "minutes": int(valid.sum())}
        if chapters:
            per = []
            for ch in chapters:
                a = int(ch["start_time"] // 60)
                b = int(np.ceil(ch["end_time"] / 60))
                dd, kk = int(dec[a:b].sum()), int(kept[a:b].sum())
                per.append({"title": ch.get("title"), "start": ch["start_time"], "end": ch["end_time"],
                            "decoded": dd, "kept": kk, "kept_fraction": kk / dd if dd else None})
            out["per_chapter"] = per
    return out
