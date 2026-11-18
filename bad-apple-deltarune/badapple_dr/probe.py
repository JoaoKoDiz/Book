"""Stage 1 -- inspect the source files and detect the active picture area.

The active area (e.g. the 4:3 game picture inside black pillarbox bars of a 16:9 upload) is used ONLY
to decide which pixels are compared during analysis. Output frames are never cropped.
"""
from __future__ import annotations

import math

import numpy as np

from .ffmpeg import grab_frames_near, probe
from .util import LOG


def detect_active_area(ffmpeg: str, info: dict, n_samples: int = 90, black_level: int = 24,
                       workers: int = 4) -> dict:
    """Find constant black borders by sampling frames spread over the WHOLE video.

    A column/row belongs to a bar when its 95th-percentile brightness stays below `black_level` in at
    least 95 % of the samples. Sampling the whole file matters: a reference that is dark during its
    first seconds would otherwise look like it had bars.
    """
    v = info["video"]
    W, H = v["width"], v["height"]
    dur = float(info.get("duration") or v.get("duration") or 0.0)
    full = {"x": 0, "y": 0, "w": W, "h": H, "method": "full-frame", "samples": 0}
    if dur <= 0 or W <= 0 or H <= 0:
        return full
    scale = min(1.0, 960.0 / W)
    sw, sh = max(2, int(round(W * scale / 2)) * 2), max(2, int(round(H * scale / 2)) * 2)
    times = np.linspace(0.01 * dur, 0.97 * dur, n_samples)
    frames = grab_frames_near(ffmpeg, info["path"], times, file_start=0.0, workers=workers,
                              pix_fmt="gray", size=(sw, sh))
    if len(frames) < max(5, n_samples // 4):
        LOG.warning("Active-area detection: only %d frames could be sampled; using the full frame.", len(frames))
        return full
    stack = np.stack([f for _, f in frames]).astype(np.uint8)  # (n, sh, sw)
    col_act = np.percentile(stack, 95, axis=1)  # (n, sw)
    row_act = np.percentile(stack, 95, axis=2)  # (n, sh)
    bar_col = (col_act < black_level).mean(axis=0) >= 0.95
    bar_row = (row_act < black_level).mean(axis=0) >= 0.95

    def span(bar):
        idx = np.where(~bar)[0]
        if len(idx) == 0:
            return 0, len(bar)
        return int(idx[0]), int(idx[-1]) + 1

    x0, x1 = span(bar_col)
    y0, y1 = span(bar_row)
    # back to native coordinates, even-aligned, slightly inside the detected area
    X0 = int(math.ceil(x0 / scale / 2.0)) * 2
    X1 = int(math.floor(x1 / scale / 2.0)) * 2
    Y0 = int(math.ceil(y0 / scale / 2.0)) * 2
    Y1 = int(math.floor(y1 / scale / 2.0)) * 2
    w, h = X1 - X0, Y1 - Y0
    if w < 0.4 * W or h < 0.4 * H:
        LOG.warning("Active-area detection gave an implausible area (%dx%d of %dx%d); using full frame.", w, h, W, H)
        full["samples"] = len(frames)
        return full
    area = {"x": X0, "y": Y0, "w": w, "h": h, "method": "auto (black-bar detection)", "samples": len(frames),
            "bars_px": {"left": X0, "right": W - X1, "top": Y0, "bottom": H - Y1}}
    if (w, h) == (W, H):
        area["method"] = "full-frame (no bars found)"
    return area


def area_from_override(crop, info: dict) -> dict:
    x, y, w, h = [int(v) for v in crop]
    v = info["video"]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > v["width"] or y + h > v["height"]:
        raise ValueError(f"crop override {crop} lies outside the {v['width']}x{v['height']} frame")
    return {"x": x, "y": y, "w": w, "h": h, "method": "manual override", "samples": 0}


def inspect_file(ffmpeg: str, ffprobe: str | None, path: str, role: str, crop_override=None,
                 n_samples: int = 90) -> dict:
    info = probe(path, ffprobe=ffprobe, ffmpeg=ffmpeg)
    v = info["video"]
    LOG.info("[%s] %s", role, info["path"])
    LOG.info("    container=%s duration=%.3fs (%s) start=%.3fs size=%.1f MB",
             info.get("format_name"), info.get("duration") or -1, _hms(info.get("duration")),
             info.get("start_time") or 0.0, info["size_bytes"] / 1e6)
    LOG.info("    video: %s %dx%d SAR %s DAR %s pix_fmt=%s fps=%s (%.3f, CFR likely=%s) frames=%s",
             v["codec"], v["width"], v["height"], v["sar"], v["dar"], v["pix_fmt"], v["fps"], v["fps_float"],
             v["cfr_likely"], v["nb_frames"] if v["nb_frames"] else "n/a")
    if info["audio"]:
        for a in info["audio"]:
            LOG.info("    audio: %s %d Hz %d ch", a["codec"], a["sample_rate"], a["channels"])
    else:
        LOG.info("    audio: none")
    if crop_override:
        info["active_area"] = area_from_override(crop_override, info)
    else:
        info["active_area"] = detect_active_area(ffmpeg, info, n_samples=n_samples)
    a = info["active_area"]
    LOG.info("    analysis area: x=%d y=%d %dx%d (%s)", a["x"], a["y"], a["w"], a["h"], a["method"])
    info["role"] = role
    return info


def _hms(sec) -> str:
    if not sec:
        return "?"
    sec = int(round(sec))
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def analysis_canvas(src_area: dict, mid_width: int) -> tuple[int, int]:
    """Size (W, H) of the mid analysis level, following the aspect ratio of the source game area."""
    W = int(mid_width)
    H = int(round(W * src_area["h"] / src_area["w"] / 2.0)) * 2
    H = max(8, H)
    return W, H
