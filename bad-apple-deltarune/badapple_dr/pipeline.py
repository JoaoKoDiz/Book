"""Runs the stages in order. Each stage stores its result in the work folder and is skipped (or
resumed) when the same command is run again, so an interruption never loses finished work."""
from __future__ import annotations

import time
from pathlib import Path

from .config import save_config
from .ffmpeg import ffmpeg_version, find_ffmpeg, find_ffprobe
from .index import SourceDB, build_index
from .probe import analysis_canvas, inspect_file
from .reference import extract_reference, load_reference
from .util import (LOG, StageTimer, atomic_write_json, file_identity, fingerprint, read_json, set_stop_file,
                   setup_logging)

STAGES = ("probe", "index", "reference", "match", "export")


def _check_inputs(cfg: dict) -> None:
    for key, label in (("reference", "Bad Apple!! reference video (--reference)"),
                       ("source", "DELTARUNE gameplay video (--source)")):
        p = cfg.get(key)
        if not p:
            raise SystemExit(f"Missing {label}.")
        if not Path(p).is_file():
            raise SystemExit(f"{label} not found: {p}")
    for key in ("export_source", "reference_audio", "source_info"):
        p = cfg.get(key)
        if p and not Path(p).is_file():
            raise SystemExit(f"--{key.replace('_', '-')} file not found: {p}")


def stage_probe(ffmpeg, ffprobe, cfg, work: Path) -> dict:
    an = cfg["analysis"]
    key = {"ref": file_identity(cfg["reference"]), "src": file_identity(cfg["source"]),
           "exp": file_identity(cfg["export_source"]) if cfg.get("export_source") else None,
           "crop_src": an.get("source_crop"), "crop_ref": an.get("reference_crop"), "n": an["active_area_samples"]}
    fp = fingerprint(key)
    cached = read_json(work / "probe.json")
    if cached and cached.get("fingerprint") == fp:
        LOG.info("Stage 1: using saved source inspection (probe.json).")
        return cached
    LOG.info("Stage 1: inspecting source files.")
    info = {"fingerprint": fp,
            "reference": inspect_file(ffmpeg, ffprobe, cfg["reference"], "reference", an.get("reference_crop"),
                                      an["active_area_samples"]),
            "source": inspect_file(ffmpeg, ffprobe, cfg["source"], "source", an.get("source_crop"),
                                   an["active_area_samples"])}
    if cfg.get("export_source"):
        exp = inspect_file(ffmpeg, ffprobe, cfg["export_source"], "export source", None, an["active_area_samples"])
        s, e = info["source"], exp
        if abs((s.get("duration") or 0) - (e.get("duration") or 0)) > 2.0:
            raise SystemExit("The export source has a different duration than the analysis source -- it must be "
                             "another copy (rendition) of the SAME video.")
        info["export_source"] = exp
    if not info["reference"]["audio"] and not cfg.get("reference_audio"):
        LOG.warning("The reference file has no audio stream; the output will be silent unless --reference-audio is given.")
    atomic_write_json(work / "probe.json", info)
    return info


def stage_reference(ffmpeg, cfg, probe_info, work: Path, mid) -> dict:
    ref_dir = work / "reference"
    key = {"v": 2, "ref": file_identity(cfg["reference"]), "start": cfg["ref_start"], "dur": cfg["ref_duration"],
           "area": probe_info["reference"]["active_area"], "mid": list(mid), "s": cfg["analysis"]["fine_scale"]}
    fp = fingerprint(key)
    meta = read_json(ref_dir / "meta.json")
    if meta and meta.get("fingerprint") == fp and (ref_dir / "frames.npz").exists():
        LOG.info("Reference: using saved frames (%d frames).", meta["frames"])
        return meta
    meta = extract_reference(ffmpeg, cfg, probe_info["reference"], mid, ref_dir)
    meta["fingerprint"] = fp
    atomic_write_json(ref_dir / "meta.json", meta)
    return meta


def run(cfg: dict, until: str = "export", stop_after_frames: int | None = None, verbose: bool = False) -> dict:
    work = Path(cfg["work_dir"]).resolve()
    out = Path(cfg["out_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    setup_logging(work, verbose)
    set_stop_file(work)
    _check_inputs(cfg)
    save_config(cfg, work / "config.json")
    ffmpeg = find_ffmpeg(cfg.get("ffmpeg") or None)
    ffprobe = find_ffprobe(cfg.get("ffprobe") or None, ffmpeg)
    LOG.info("ffmpeg: %s (version %s)", ffmpeg, ".".join(map(str, ffmpeg_version(ffmpeg))))
    LOG.info("ffprobe: %s", ffprobe or "not found (falling back to parsing `ffmpeg -i`)")
    timer = StageTimer(work)
    result: dict = {"work_dir": str(work), "out_dir": str(out)}

    t = time.time()
    probe_info = stage_probe(ffmpeg, ffprobe, cfg, work)
    timer.add("stage1_probe", time.time() - t)
    mid = analysis_canvas(probe_info["source"]["active_area"], cfg["analysis"]["mid_width"])
    result["analysis_canvas"] = mid
    if until == "probe":
        return result

    t = time.time()
    LOG.info("Stage 2/3: building the source database (one sequential pass over the whole video).")
    st = build_index(ffmpeg, cfg, probe_info["source"], work / "db", mid, stop_after_frames=stop_after_frames)
    timer.add("stage2_index", time.time() - t)
    result["index_state"] = st
    if not st.get("complete"):
        LOG.info("Index not complete yet (stopped early). Run the same command again to continue.")
        return result
    if until == "index":
        return result

    t = time.time()
    ref_meta = stage_reference(ffmpeg, cfg, probe_info, work, mid)
    timer.add("reference", time.time() - t)
    if until == "reference":
        return result

    from .match import Matcher

    t = time.time()
    db = SourceDB(work / "db").open()
    ref = load_reference(work / "reference")
    matcher = Matcher(ffmpeg, cfg, probe_info["source"], db, ref, work)
    pools, sel = matcher.run()
    timer.add("stages4to7_match", time.time() - t)
    result["mean_similarity"] = sel["mean_total"]
    if until == "match":
        return result

    from .export import export_all

    t = time.time()
    timings = dict(timer.data)
    report = export_all(ffmpeg, ffprobe, cfg, probe_info, db, ref, ref_meta, pools, sel, matcher, out, work, timings)
    timer.add("export", time.time() - t)
    result["report"] = report
    LOG.info("Done. Outputs in %s (total processing time %.1f min).", out, timer.total() / 60.0)
    return result
