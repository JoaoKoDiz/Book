"""Configuration: defaults, JSON loading and CLI overrides.

Every tunable used by the pipeline lives here so that the full set of weights/thresholds of a run is
recorded in work/config.json and in the quality report.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    # ---- inputs / outputs -------------------------------------------------------------------
    "reference": "",          # Bad Apple!! video file (video + audio)
    "reference_audio": "",    # optional separate audio file for the reference (same timeline)
    "source": "",             # DELTARUNE walkthrough used for analysis (e.g. 480p30)
    "export_source": "",      # optional higher-quality copy of the SAME video used to export frames
    "source_info": "",        # optional yt-dlp .info.json (chapter names in the CSV)
    "work_dir": "work",
    "out_dir": "output",
    "ffmpeg": "",
    "ffprobe": "",
    # ---- reference window ------------------------------------------------------------------
    "ref_start": 0.0,         # seconds from the start of the reference file (0 = literal start)
    "ref_duration": 10.0,
    # ---- analysis geometry -----------------------------------------------------------------
    "analysis": {
        "mid_width": 64,              # database thumbnails: 64 x 48 for a 4:3 game area
        "fine_scale": 2,              # fine level = 2 x mid (128 x 96)
        "active_area_samples": 90,    # frames sampled to detect black bars
        "source_crop": None,          # [x, y, w, h] override of the detected game area (analysis only)
        "reference_crop": None,       # [x, y, w, h] override for the reference
    },
    # ---- stage 2: source database (adaptive sampling) ----------------------------------------
    "index": {
        "keep_threshold": 0.010,      # keep a frame when it differs this much (mean abs, 0..1) from the last kept one
        "max_gap": 1.0,               # ...or when this many seconds passed since the last kept frame
        "cut_threshold": 0.12,        # frame-to-frame change that counts as a cut / flash
        "cut_window": 3,              # frames kept densely on both sides of a cut
        "dup_limit": 3,               # max copies of an identical (quantised) thumbnail in the whole database
        "threads": 0,                 # ffmpeg decoder threads (0 = auto)
        "hwaccel": "none",            # "none", "auto", "d3d11va", "dxva2", "cuda", ...
        "commit_frames": 6000,        # checkpoint at least every N decoded frames
        "commit_seconds": 30.0,       # ...or every N seconds of wall-clock time
    },
    # ---- stage 4: exact coarse retrieval over the whole database ----------------------------
    "retrieval": {
        "coarse_topk": 6000,          # best coarse candidates kept per target (exact search)
        "candidates": 400,            # after temporal diversification -> detailed comparison
        "nms_window": 0.5,            # seconds; diversification window
        "max_per_window": 3,          # at most this many candidates per window
        "chunk": 4096,                # database rows per BLAS chunk (~0.5 GB RAM peak)
    },
    # ---- stage 5: detailed comparison ---------------------------------------------------------
    "rerank": {
        "fine_topn": 32,              # candidates per target re-scored at the fine level (frames re-decoded)
        "workers": 4,                 # parallel ffmpeg processes for frame fetching
    },
    # ---- stage 6: temporal optimisation ----------------------------------------------------
    "temporal": {
        "lambda_rhythm": 0.30,        # weight: |output change - reference change| between consecutive frames
        "lambda_jump": 0.03,          # weight: jumping to an unrelated part of the gameplay
        "max_drop": 0.04,             # a frame may lose at most this much similarity for coherence
        "scene_seconds": 3.0,         # candidates closer than this in the source count as same scene
        "ref_cut": 0.25,              # reference change above which jumps are free (reference cut)
    },
    # ---- stage 7: targeted refinement -------------------------------------------------------
    "refine": {
        "rounds": 2,
        "window": 1.0,                # seconds decoded densely on each side of a promising candidate
        "per_target": 4,              # promising candidates per target whose neighbourhood is decoded
        "neighbours": 2,              # also score window frames against targets t-2..t+2
        "prefilter": 24,              # per target, window frames fully scored after the coarse prefilter
        "worst_fraction": 0.25,       # worst targets get a wider second-chance search
        "extra_candidates": 1500,     # second chance: candidates kept after diversification
        "extra_fine": 64,             # second chance: candidates re-scored at the fine level
        "max_frames_per_round": 150000,
        "min_gain": 0.0005,           # stop when the mean similarity improves less than this
    },
    # ---- optional semantic features --------------------------------------------------------------
    "semantic": {
        "mode": "auto",               # "none" | "auto" (use if torch+transformers+model available) | "on"
        "model": "facebook/dinov2-small",
        "device": "auto",             # "auto" | "cpu" | "cuda"
        "batch": 32,
        "image_width": 224,
    },
    # ---- similarity weights --------------------------------------------------------------------
    "weights": {
        "composition": 0.50,
        "structural": 0.40,
        "semantic": 0.10,
        "composition_parts": {"tone": 0.70, "balance": 0.30},
        # tone layout levels: 24x32, 12x16, 6x8, 3x4 (for a 48x64 mid canvas)
        "tone_levels": [0.25, 0.30, 0.25, 0.20],
        "structural_parts": {
            "ssim_mid": 0.20, "zncc": 0.15, "biou_mid": 0.20, "chamfer_mid": 0.15, "hog": 0.10,
            "ssim_fine": 0.05, "biou_fine": 0.05, "chamfer_fine": 0.10,
        },
        "coarse": {"tone16": 0.30, "tone32": 0.20, "zncc32": 0.25, "dice32": 0.25},
        "uniform_std": 0.04,          # reference frames flatter than this are treated as uniform
    },
    # ---- export ---------------------------------------------------------------------------
    "export": {
        "output_size": "source",      # "source" (native size of the export source) or "WIDTHxHEIGHT"
        "crf": 14,
        "preset": "slow",
        "audio_bitrate": "192k",
        "comparison_height": 360,
        "lossless": False,            # True -> x264 lossless (qp 0); bigger files, less compatible
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        unknown = set(user) - set(DEFAULTS) - {"_comment"}
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        cfg = deep_merge(cfg, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def set_dotted(cfg: dict, dotted: str, value) -> None:
    """Set cfg['a']['b'] from 'a.b'. Values are parsed as JSON when possible ("0.5" -> 0.5)."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            raise KeyError(f"unknown config section '{k}' in '{dotted}'")
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"unknown config key '{dotted}'")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    d[keys[-1]] = value


def save_config(cfg: dict, path: str | Path) -> None:
    from .util import atomic_write_json

    atomic_write_json(path, cfg)
