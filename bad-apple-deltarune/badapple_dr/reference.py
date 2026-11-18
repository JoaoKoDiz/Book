"""Reference (Bad Apple!!) frame extraction.

The window is [ref_start, ref_start + ref_duration) measured from the start of the file
(ref_start = 0 -> the literal first seconds, including any intro/padding the upload contains).
Every reference frame whose exact timestamp falls inside the window becomes one output frame.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import features as F
from .ffmpeg import FrameReader
from .util import LOG, atomic_save_npz, fmt_timecode


def extract_reference(ffmpeg: str, cfg: dict, ref_info: dict, canvas_mid: tuple[int, int], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    v = ref_info["video"]
    fps = Fraction(v["fps"])
    file_start = float(ref_info.get("start_time") or 0.0)
    # Time 0 of the reference = its FIRST VIDEO FRAME (containers often start earlier because of audio
    # encoder priming, e.g. -0.021 s). Video frames and the audio window are both anchored to it.
    origin = first_frame_pts(ffmpeg, ref_info["path"])
    t0 = origin + float(cfg["ref_start"])
    t1 = t0 + float(cfg["ref_duration"])
    area = ref_info["active_area"]
    s = int(cfg["analysis"]["fine_scale"])
    Wf, Hf = canvas_mid[0] * s, canvas_mid[1] * s
    seek = (t0 - file_start) - 1.0
    pts_l, fine_l = [], []
    with FrameReader(ffmpeg, ref_info["path"], pix_fmt="gray", crop=(area["x"], area["y"], area["w"], area["h"]),
                     fit=(Wf, Hf), seek=seek if seek > 0 else None) as rd:
        for pts, fr in rd:
            if pts < t0 - 1e-6:
                continue
            if pts >= t1 - 1e-9:
                break
            pts_l.append(pts)
            fine_l.append(fr.copy())
    if not pts_l:
        raise RuntimeError("No reference frames found in the requested window -- check --ref-start / the file.")
    pts = np.array(pts_l, np.float64)
    fine = np.stack(fine_l)
    mid = np.clip(np.round(F.pool2(fine.astype(np.float32))), 0, 255).astype(np.uint8)
    expected = float(cfg["ref_duration"]) * float(fps)
    n = len(pts)
    LOG.info("Reference: %d frames in [%s, %s) at %s fps (%.3f expected from duration x fps); first video frame "
             "at %.3fs in the file timeline.", n, fmt_timecode(t0 - origin), fmt_timecode(t1 - origin), v["fps"],
             expected, origin)
    gaps = np.diff(pts)
    irregular = bool(len(gaps) and (np.abs(gaps - 1.0 / float(fps)) > 0.5 / float(fps)).any())
    if irregular:
        LOG.warning("Reference frame timestamps are irregular (VFR?); output uses a constant %s fps.", v["fps"])
    atomic_save_npz(out_dir / "frames.npz", pts=pts, rel_t=pts - origin, fine=fine, mid=mid,
                    fps_num=fps.numerator, fps_den=fps.denominator)
    sheet = contact_sheet([fine[i] for i in range(n)], [f"{i} {pts[i] - origin:.2f}s" for i in range(n)],
                          step=max(1, n // 48))
    cv2.imwrite(str(out_dir / "reference_contact_sheet.png"), sheet)
    meta = {"frames": n, "fps": v["fps"], "fps_float": float(fps), "window_start": t0 - origin,
            "window_end": t1 - origin, "window_abs_start": t0, "origin_pts": origin, "file_start": file_start,
            "audio_seek": t0 - file_start, "expected_frames": expected,
            "irregular_timestamps": irregular, "fine_size": [Wf, Hf], "mid_size": list(canvas_mid),
            "active_area": area, "first_pts": float(pts[0]), "last_pts": float(pts[-1])}
    return meta


def first_frame_pts(ffmpeg: str, path: str) -> float:
    with FrameReader(ffmpeg, path, pix_fmt="gray", size=(16, 16), max_frames=1) as rd:
        pts, _ = rd.read_batch(1)
    if not len(pts):
        raise RuntimeError(f"Could not decode the first frame of {path}")
    return float(pts[0])


def load_reference(ref_dir: Path) -> dict:
    d = np.load(ref_dir / "frames.npz")
    return {k: d[k] for k in d.files}


def contact_sheet(images, labels, step=1, cols=8, cell_w=192) -> np.ndarray:
    """Grid of thumbnails (grey or BGR) with small labels (diagnostic image only)."""
    sel = list(range(0, len(images), step))
    if not sel:
        return np.zeros((10, 10, 3), np.uint8)
    h0, w0 = images[0].shape[:2]
    cell_h = int(round(cell_w * h0 / w0))
    rows = int(math.ceil(len(sel) / cols))
    sheet = np.full((rows * (cell_h + 18), cols * cell_w, 3), 40, np.uint8)
    for k, i in enumerate(sel):
        im = images[i]
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        y, x = r * (cell_h + 18), c * cell_w
        sheet[y:y + cell_h, x:x + cell_w] = im
        cv2.putText(sheet, str(labels[i])[:30], (x + 3, y + cell_h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return sheet
