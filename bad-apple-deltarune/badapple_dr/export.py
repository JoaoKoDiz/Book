"""Export: final reconstruction, frame mapping CSV, diagnostic comparison video, quality report.

The final video is assembled from frames decoded at their exact timestamps from the gameplay video
in its native pixel format and size and passed to the encoder unchanged (no crop, no colour
changes, no overlays). Optional uniform resizing + letterboxing is available via export.output_size.
After encoding, the output is decoded again and every frame is compared with the source frame it
claims to be (traceability check); the audio offset is measured by cross-correlation.
"""
from __future__ import annotations

import csv
import math
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from . import features as F
from .ffmpeg import _NO_WINDOW, FrameReader, fetch_frames, probe, run_ffmpeg
from .match import ORIGIN_NAMES, mid_from_fine
from .reference import contact_sheet
from .util import LOG, Progress, atomic_write_json, fmt_timecode

FINAL_NAME = "Bad_Apple_Deltarune_10s.mp4"
CSV_NAME = "frame_mapping.csv"
COMPARE_NAME = "comparison_10s.mp4"
REPORT_NAME = "quality_report.json"


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def _color_args(v: dict) -> list[str]:
    args = []
    if v.get("color_primaries") and v["color_primaries"] != "unknown":
        args += ["-color_primaries", v["color_primaries"]]
    if v.get("color_transfer") and v["color_transfer"] != "unknown":
        args += ["-color_trc", v["color_transfer"]]
    if v.get("color_space") and v["color_space"] != "unknown":
        args += ["-colorspace", v["color_space"]]
    if v.get("color_range") in ("tv", "pc"):
        args += ["-color_range", v["color_range"]]
    return args


def chapters_for(cfg: dict, src_path: str) -> list:
    """Chapters from a yt-dlp .info.json (explicit option, or next to the source file)."""
    import json

    cands = []
    if cfg.get("source_info"):
        cands.append(Path(cfg["source_info"]))
    p = Path(src_path)
    cands += [p.with_suffix(".info.json"), p.with_name(p.stem.split(".")[0] + ".info.json")]
    for c in cands:
        if c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    ch = json.load(f).get("chapters") or []
                return [{"start_time": float(x["start_time"]), "end_time": float(x["end_time"]),
                         "title": x.get("title", "")} for x in ch]
            except Exception as e:
                LOG.warning("Could not read chapters from %s: %s", c, e)
    return []


def chapter_of(chs: list, t: float) -> str:
    for c in chs:
        if c["start_time"] <= t < c["end_time"]:
            return c["title"]
    return ""


def _audio_source(cfg: dict, ref_info: dict, ref_meta: dict) -> tuple[str | None, float]:
    """(audio file, input seek in seconds relative to that file's start).

    For the reference file the seek is computed from the first video frame, so the audio window covers
    exactly the same absolute time span as the selected reference frames (container start offsets such
    as AAC priming are compensated). A separate audio file is assumed to start with the video.
    """
    if cfg.get("reference_audio"):
        return cfg["reference_audio"], float(cfg["ref_start"])
    if ref_info.get("audio"):
        return ref_info["path"], max(0.0, float(ref_meta.get("audio_seek", cfg["ref_start"])))
    return None, 0.0


def _decode_audio(ffmpeg: str, path: str, start: float, dur: float, rate: int = 8000) -> np.ndarray:
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-t", f"{dur:.6f}", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    return np.frombuffer(r.stdout, np.float32)


def audio_offset_ms(ffmpeg: str, a_path: str, a_start: float, b_path: str, dur: float) -> float | None:
    """Lag (ms) of b's audio relative to the reference audio window; ~0 when in sync."""
    rate = 8000
    a = _decode_audio(ffmpeg, a_path, a_start, dur, rate)
    b = _decode_audio(ffmpeg, b_path, 0.0, dur, rate)
    n = min(len(a), len(b))
    if n < rate // 2 or np.std(a[:n]) < 1e-5 or np.std(b[:n]) < 1e-5:
        return None
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    m = 1 << int(math.ceil(math.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(b, m) * np.conj(np.fft.rfft(a, m)), m)
    maxlag = int(0.5 * rate)
    lags = np.r_[np.arange(0, maxlag), np.arange(-maxlag, 0)]
    vals = np.r_[xc[:maxlag], xc[-maxlag:]]
    return float(lags[int(np.argmax(vals))] * 1000.0 / rate)


def export_all(ffmpeg: str, ffprobe: str | None, cfg: dict, probe_info: dict, db, ref: dict, ref_meta: dict,
               pools: list[dict], sel: dict, matcher, out_dir: Path, work: Path, timings: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_export = time.time()
    ecfg = cfg["export"]
    src = probe_info["source"]
    exp = probe_info.get("export_source") or src
    ref_info = probe_info["reference"]
    same_file = exp["path"] == src["path"]
    T = len(pools)
    path = sel["path"]
    chosen = [{k: pools[t][k][path[t]] for k in pools[t]} for t in range(T)]
    fps_ref = Fraction(ref_meta["fps"])
    src_fps = float(src["video"]["fps_float"])
    src_start = float(src.get("start_time") or 0.0)
    exp_v = exp["video"]
    exp_fps = float(exp_v["fps_float"])
    exp_start = float(exp.get("start_time") or 0.0)
    W, H = int(exp_v["width"]), int(exp_v["height"])
    Hf, Wf = matcher.fine_shape
    ea = exp["active_area"]
    # ------------------------------------------------------------------ fetch native frames
    uniq = sorted({float(c["pts"]) for c in chosen})
    LOG.info("Export: fetching %d unique gameplay frames (native %dx%d) from %s", len(uniq), W, H, Path(exp["path"]).name)
    native: dict[float, tuple[float, np.ndarray]] = {}
    trace: dict[float, dict] = {}
    analysis_kw = {"pix_fmt": "gray", "size": (Wf, Hf), "crop": (ea["x"], ea["y"], ea["w"], ea["h"])}
    if same_file:
        res = fetch_frames(ffmpeg, exp["path"], uniq, file_start=exp_start, fps=exp_fps, workers=matcher.workers,
                           tolerance=matcher._tol(), pix_fmt="yuv420p", native_size=(W, H))
        chk = fetch_frames(ffmpeg, exp["path"], uniq, file_start=exp_start, fps=exp_fps, workers=matcher.workers,
                           tolerance=matcher._tol(), **analysis_kw)
        for p in uniq:
            if p not in res:
                raise RuntimeError(f"Could not decode the selected frame at {p:.6f}s for export.")
            native[p] = res[p]
            n = int(next(c["n"] for c in chosen if float(c["pts"]) == p))
            a_img = matcher.store.get_many([n])[0]
            mad = float(np.abs(chk[p][1].astype(np.int16) - a_img.astype(np.int16)).mean()) if p in chk else float("nan")
            trace[p] = {"export_pts": res[p][0], "mad": mad}
    else:
        # different rendition of the same video: pick, among the frames near the timestamp, the one
        # whose analysis image matches the analysed frame best
        near = []
        for p in uniq:
            near += [p + k / exp_fps for k in range(-2, 3)]
        chk = fetch_frames(ffmpeg, exp["path"], near, file_start=exp_start, fps=exp_fps, workers=matcher.workers,
                           tolerance=0.5 / exp_fps, **analysis_kw)
        pick = {}
        for p in uniq:
            n = int(next(c["n"] for c in chosen if float(c["pts"]) == p))
            a_img = matcher.store.get_many([n])[0].astype(np.int16)
            best = None
            for k in range(-2, 3):
                q = p + k / exp_fps
                if q in chk:
                    mad = float(np.abs(chk[q][1].astype(np.int16) - a_img).mean())
                    if best is None or mad < best[0]:
                        best = (mad, chk[q][0])
            if best is None:
                raise RuntimeError(f"Export source has no frame near {p:.3f}s")
            pick[p] = best
        res = fetch_frames(ffmpeg, exp["path"], [v[1] for v in pick.values()], file_start=exp_start, fps=exp_fps,
                           workers=matcher.workers, tolerance=matcher._tol(), pix_fmt="yuv420p", native_size=(W, H))
        for p, (mad, ep) in pick.items():
            native[p] = res[ep]
            trace[p] = {"export_pts": res[ep][0], "mad": mad}
            if mad > 8:
                LOG.warning("Export rendition frame near %.3fs differs from the analysed frame (MAD %.1f).", p, mad)
    # ------------------------------------------------------------------ encode final video
    out_size = str(ecfg.get("output_size", "source")).lower()
    vf = []
    OW, OH = W, H
    if out_size not in ("source", "native", ""):
        OW, OH = [int(x) for x in out_size.lower().split("x")]
        vf = [f"scale={OW}:{OH}:force_original_aspect_ratio=decrease:flags=lanczos",
              f"pad={OW}:{OH}:(ow-iw)/2:(oh-ih)/2:color=black"]
    final = out_dir / FINAL_NAME
    tmp = out_dir / (FINAL_NAME + ".part.mp4")
    a_path, a_start = _audio_source(cfg, ref_info, ref_meta)
    ref_origin = float(ref_meta.get("origin_pts", ref_info.get("start_time") or 0.0))
    dur_video = T / float(fps_ref)
    dur = float(cfg["ref_duration"])
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-framerate", str(fps_ref), "-i", "pipe:0"]
    if a_path:
        if a_start > 1e-6:
            cmd += ["-ss", f"{a_start:.6f}"]
        cmd += ["-t", f"{dur:.6f}", "-i", a_path, "-map", "0:v:0", "-map", "1:a:0"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if ecfg.get("lossless"):
        cmd += ["-c:v", "libx264", "-preset", "veryslow", "-qp", "0", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "libx264", "-preset", str(ecfg.get("preset", "slow")), "-crf", str(int(ecfg.get("crf", 14))),
                "-pix_fmt", "yuv420p"]
    cmd += _color_args(exp_v)
    if a_path:
        cmd += ["-c:a", "aac", "-b:a", str(ecfg.get("audio_bitrate", "192k"))]
    cmd += ["-movflags", "+faststart", "-t", f"{max(dur, dur_video):.6f}", str(tmp)]
    LOG.info("Encoding %s (%d frames @ %s fps = %.3fs, %dx%d%s)", FINAL_NAME, T, fps_ref, dur_video, OW, OH,
             ", with reference audio" if a_path else ", NO audio track")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    try:
        for t in range(T):
            proc.stdin.write(native[float(chosen[t]["pts"])][1].tobytes())  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")  # type: ignore[union-attr]
    if proc.wait() != 0:
        raise RuntimeError("Encoding failed:\n" + err[-2000:])
    tmp.replace(final)
    # ------------------------------------------------------------------ comparison video
    compare = out_dir / COMPARE_NAME
    _comparison(ffmpeg, cfg, ref_info, ref_meta, exp, chosen, pools, sel, compare, a_path, a_start, fps_ref, matcher)
    # ------------------------------------------------------------------ verification
    verify = verify_output(ffmpeg, ffprobe, final, chosen, matcher, exp, T, fps_ref, dur, a_path, a_start,
                           same_file, vf)
    # ------------------------------------------------------------------ CSV
    chs = chapters_for(cfg, src["path"])
    ref_pts = ref["pts"]
    with open(out_dir / CSV_NAME, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["output_frame_index", "output_timestamp_s", "reference_frame_index", "reference_timestamp_s",
                     "source_frame_index", "source_timestamp_s", "source_timecode", "source_chapter",
                     "total_similarity", "composition_similarity", "structural_similarity", "semantic_similarity",
                     "temporal_coherence", "tone_layout", "balance", "ssim_mid", "zncc", "figure_ground_iou_mid",
                     "contour_chamfer_mid", "gradient_orientation", "ssim_fine", "figure_ground_iou_fine",
                     "contour_chamfer_fine", "uniform_reference_frame", "found_by", "in_database",
                     "export_timestamp_s", "trace_mad", "output_mad"])
        for t in range(T):
            c = chosen[t]
            p = float(c["pts"])
            rel = p - src_start
            tr = trace[p]

            def g(k):
                v = float(c[k])
                return "" if not np.isfinite(v) else f"{v:.5f}"

            wr.writerow([t, f"{t / float(fps_ref):.6f}", int(round((ref_pts[t] - ref_origin) * float(fps_ref))),
                         f"{ref_pts[t] - ref_origin:.6f}", int(c["n"]), f"{rel:.6f}", fmt_timecode(rel),
                         chapter_of(chs, rel), g("total"), g("composition"), g("structural"), g("semantic"),
                         f"{float(sel['coherence'][t]):.5f}", g("tone"), g("balance"), g("ssim_mid"), g("zncc"),
                         g("biou_mid"), g("chamfer_mid"), g("hog"), g("ssim_fine"), g("biou_fine"), g("chamfer_fine"),
                         int(bool(matcher.ts.uniform[t])), ORIGIN_NAMES.get(int(c["origin"]), "?"),
                         int(int(c["row"]) >= 0), f"{tr['export_pts'] - exp_start:.6f}", f"{tr['mad']:.3f}",
                         "" if verify["per_frame_mad"] is None else f"{verify['per_frame_mad'][t]:.3f}"])
    # ------------------------------------------------------------------ contact sheet
    try:
        _pairs_sheet(ref, chosen, matcher, out_dir / "contact_sheet.png", src_start)
    except Exception as e:  # diagnostic only
        LOG.warning("contact sheet failed: %s", e)
    timings["export"] = time.time() - t_export
    report = build_report(cfg, probe_info, db, ref_meta, pools, sel, chosen, matcher, verify, timings, chs, trace)
    atomic_write_json(out_dir / REPORT_NAME, report)
    LOG.info("Wrote %s, %s, %s, %s", final.name, CSV_NAME, COMPARE_NAME, REPORT_NAME)
    return report


def _comparison(ffmpeg, cfg, ref_info, ref_meta, exp, chosen, pools, sel, out_path, a_path, a_start, fps_ref,
                matcher):
    Hc = int(cfg["export"].get("comparison_height", 360))
    rv, ev = ref_info["video"], exp["video"]
    Wr = _even(Hc * rv["width"] / rv["height"])
    We = _even(Hc * ev["width"] / ev["height"])
    strip = 40
    file_start = float(ref_info.get("start_time") or 0.0)
    t0 = float(ref_meta.get("window_abs_start", file_start + float(cfg["ref_start"])))
    t1 = t0 + float(cfg["ref_duration"])
    refs = []
    seek = (t0 - file_start) - 1.0
    with FrameReader(ffmpeg, ref_info["path"], pix_fmt="bgr24", size=(Wr, Hc), seek=seek if seek > 0 else None) as rd:
        for pts, fr in rd:
            if pts < t0 - 1e-6:
                continue
            if pts >= t1 - 1e-9 or len(refs) >= len(chosen):
                break
            refs.append(fr.copy())
    exp_start = float(exp.get("start_time") or 0.0)
    uniq = sorted({float(c["pts"]) for c in chosen})
    res = fetch_frames(ffmpeg, exp["path"], uniq, file_start=exp_start, fps=float(ev["fps_float"]),
                       workers=matcher.workers, tolerance=0.5 / float(ev["fps_float"]), pix_fmt="bgr24", size=(We, Hc))
    Wt = Wr + We
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{Wt}x{Hc + strip}", "-framerate", str(fps_ref), "-i", "pipe:0"]
    if a_path:
        if a_start > 1e-6:
            cmd += ["-ss", f"{a_start:.6f}"]
        cmd += ["-t", f"{float(cfg['ref_duration']):.6f}", "-i", a_path, "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    src_start = float(exp.get("start_time") or 0.0)
    for t, c in enumerate(chosen):
        canvas = np.zeros((Hc + strip, Wt, 3), np.uint8)
        if t < len(refs):
            canvas[:Hc, :Wr] = refs[t]
        fr = res.get(float(c["pts"]))
        if fr is not None:
            canvas[:Hc, Wr:] = fr[1]
        txt1 = (f"ref #{t}  {t / float(fps_ref):6.3f}s   |   gameplay frame #{int(c['n'])}  "
                f"{fmt_timecode(float(c['pts']) - src_start)}")
        txt2 = (f"total {float(c['total']):.3f}  composition {float(c['composition']):.3f}  "
                f"structure {float(c['structural']):.3f}  coherence {float(sel['coherence'][t]):.2f}")
        cv2.putText(canvas, txt1, (6, Hc + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt2, (6, Hc + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 220, 255), 1, cv2.LINE_AA)
        proc.stdin.write(canvas.tobytes())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
    err = proc.stderr.read().decode("utf-8", "replace")  # type: ignore[union-attr]
    if proc.wait() != 0:
        raise RuntimeError("Encoding the comparison video failed:\n" + err[-2000:])


def verify_output(ffmpeg, ffprobe, final: Path, chosen, matcher, exp, T, fps_ref, dur, a_path, a_start,
                  same_file, vf) -> dict:
    """Decode the produced MP4 and check duration, frame count, frame identity and audio sync."""
    info = probe(str(final), ffprobe=ffprobe, ffmpeg=ffmpeg)
    v = info["video"]
    out = {"output_duration_s": info.get("duration"), "video_stream_duration_s": v.get("duration"),
           "output_fps": v["fps"], "output_size": [v["width"], v["height"]], "output_codec": v["codec"],
           "has_audio": bool(info["audio"]),
           "audio_duration_s": info["audio"][0].get("duration") if info["audio"] else None}
    ea = exp["active_area"]
    Hf, Wf = matcher.fine_shape
    per = None
    if not vf:  # same geometry as the export source -> compare against the analysed images
        pts_l, frames = [], []
        with FrameReader(ffmpeg, str(final), pix_fmt="gray", size=(Wf, Hf),
                         crop=(ea["x"], ea["y"], ea["w"], ea["h"])) as rd:
            for p, fr in rd:
                pts_l.append(p)
                frames.append(fr.copy())
        out["decoded_output_frames"] = len(frames)
        ana = matcher.store.get_many([int(c["n"]) for c in chosen])
        per = [float(np.abs(frames[t].astype(np.int16) - ana[t].astype(np.int16)).mean()) if t < len(frames)
               else float("nan") for t in range(T)]
        # identity check: each output frame must be closer to its own source frame than to the others
        okc = 0
        if len(frames) >= T:
            fm = mid_from_fine(np.stack(frames[:T])).astype(np.float32)
            am = mid_from_fine(ana).astype(np.float32)
            for t in range(T):
                d = np.abs(am - fm[t][None]).mean(axis=(1, 2))
                okc += int(d[t] <= d.min() + 1e-6)
        out["frames_identical_to_selected_source_frames"] = okc
        out["per_frame_mad_mean"] = float(np.nanmean(per))
        out["per_frame_mad_max"] = float(np.nanmax(per))
    else:  # resized output: frame identity cannot be compared pixel-wise, but the frame count can
        with FrameReader(ffmpeg, str(final), pix_fmt="gray", size=(16, 16)) as rd:
            out["decoded_output_frames"] = sum(1 for _ in rd)
    out["per_frame_mad"] = per
    out["expected_frames"] = T
    out["frame_count_ok"] = (out["decoded_output_frames"] == T) if out["decoded_output_frames"] is not None else None
    if a_path:
        out["audio_offset_ms"] = audio_offset_ms(ffmpeg, a_path, a_start, str(final), dur)
    return out


def _pairs_sheet(ref, chosen, matcher, path: Path, src_start: float):
    fine = matcher.store.get_many([int(c["n"]) for c in chosen])
    imgs, labels = [], []
    step = max(1, len(chosen) // 24)
    for t in range(0, len(chosen), step):
        imgs += [ref["fine"][t], fine[t]]
        labels += [f"ref {t}", f"#{int(chosen[t]['n'])} {fmt_timecode(float(chosen[t]['pts']) - src_start)}"]
    sheet = contact_sheet(imgs, labels, step=1, cols=8, cell_w=176)
    cv2.imwrite(str(path), sheet)


def build_report(cfg, probe_info, db, ref_meta, pools, sel, chosen, matcher, verify, timings, chs, trace) -> dict:
    from .index import database_report

    T = len(chosen)
    tot = np.array([float(c["total"]) for c in chosen])
    comp = np.array([float(c["composition"]) for c in chosen])
    struct = np.array([float(c["structural"]) for c in chosen])
    sem = np.array([float(c["semantic"]) for c in chosen])
    src_start = float(probe_info["source"].get("start_time") or 0.0)
    worst = np.argsort(tot)[: min(15, T)]
    st = matcher.stats
    dbr = database_report(db, chs)
    rel = [float(c["pts"]) - src_start for c in chosen]
    span = [min(rel), max(rel)] if rel else None
    chapters_used = sorted({chapter_of(chs, r) for r in rel} - {""}) if chs else []
    uniq = len({int(c["n"]) for c in chosen})
    v_ok = verify.get("frame_count_ok")
    return {
        "tool": f"badapple_dr {__version__}",
        "sources": {k: {"path": v["path"], "duration_s": v.get("duration"), "video": v["video"],
                        "audio": v.get("audio"), "analysis_area": v.get("active_area")}
                    for k, v in probe_info.items() if isinstance(v, dict) and "video" in v},
        "reference_window": {"start_s": ref_meta["window_start"], "end_s": ref_meta["window_end"],
                             "fps": ref_meta["fps"], "frames": ref_meta["frames"],
                             "expected_frames_from_duration": ref_meta["expected_frames"],
                             "timing_note": _timing_note(ref_meta)},
        "number_of_reference_frames": T,
        "source_frames": {
            "decoded_and_analysed_in_full_pass": dbr["frames_decoded_and_analysed"],
            "stored_in_search_database": dbr["frames_in_database"],
            "compared_at_fine_level": int(st.get("fine_compared_s5", 0) + st.get("fine_compared_s7", 0)),
            "re_decoded_for_fine_comparison": int(st.get("fine_fetched_s5", 0) + st.get("fine_fetched_s7", 0)),
            "decoded_densely_in_refinement": st.get("dense_frames_decoded"),
            "dense_frame_target_pairs_scored": st.get("dense_frames_scored"),
        },
        "unique_source_frames_selected": uniq,
        "selected_source_span_s": span,
        "chapters_used": chapters_used,
        "similarity": {
            "mean_total": float(tot.mean()), "median_total": float(np.median(tot)), "min_total": float(tot.min()),
            "max_total": float(tot.max()), "mean_composition": float(comp.mean()),
            "mean_structural": float(struct.mean()),
            "mean_semantic": float(np.nanmean(sem)) if np.isfinite(sem).any() else None,
            "mean_temporal_coherence": float(np.mean(sel["coherence"])),
            "mean_best_possible_per_frame": sel["mean_best"],
            "accuracy_cost_of_temporal_optimisation": float(sel["mean_best"] - tot.mean()),
            "uniform_reference_frames": int(matcher.ts.uniform.sum()),
        },
        "worst_matching_frames": [
            {"output_frame_index": int(t), "reference_timestamp_s": float(t / float(Fraction(ref_meta["fps"]))),
             "total_similarity": float(tot[t]), "source_frame_index": int(chosen[t]["n"]),
             "source_timecode": fmt_timecode(float(chosen[t]["pts"]) - src_start)} for t in worst],
        "found_by": {name: int(sum(1 for c in chosen if int(c["origin"]) == o)) for o, name in ORIGIN_NAMES.items()},
        "processing_time_s": {**{k: round(float(v), 2) for k, v in timings.items()},
                              "total": round(float(sum(timings.values())), 2)},
        "sampling_density": dbr,
        "refinement_history": st.get("refine_history"),
        "verification": {k: v for k, v in verify.items() if k != "per_frame_mad"},
        "verification_passed": bool(v_ok) if v_ok is not None else None,
        "traceability": {
            "every_output_frame_is_a_decoded_source_frame": True,
            "max_reanalysis_mad": float(np.nanmax([x["mad"] for x in trace.values()])) if trace else None,
            "note": "trace_mad: the frame at the exported timestamp, decoded again through the analysis path, "
                    "compared with the analysed image (0 = identical). output_mad: the final MP4 decoded "
                    "and compared with the analysed image (small values come from video compression only).",
        },
        "methodology": methodology(cfg, matcher),
        "configuration": {k: cfg[k] for k in ("analysis", "index", "retrieval", "rerank", "temporal", "refine",
                                              "semantic", "weights", "export")},
        "important_limitations": limitations(matcher),
        "score_disclaimer": ("All similarity scores are specific to this algorithm (weighted combinations of the "
                             "measures listed under methodology, scaled to 0..1). They are NOT percentages of "
                             "human-perceived similarity and are only comparable within this configuration."),
    }


def _timing_note(ref_meta) -> str:
    fps = Fraction(ref_meta["fps"])
    n = ref_meta["frames"]
    d = n / fps
    if d == Fraction(ref_meta["window_end"] - ref_meta["window_start"]).limit_denominator(1000):
        return f"{n} frames at {fps} fps = exactly {float(d):.3f} s."
    return (f"{n} reference frames have timestamps inside the window; at a constant {fps} fps they span "
            f"{float(d):.4f} s. The audio track is cut to exactly the window length.")


def methodology(cfg, matcher) -> dict:
    w = cfg["weights"]
    return {
        "pipeline": [
            "1 inspect files (ffprobe), detect the game area inside black bars (analysis only)",
            "2 decode the ENTIRE gameplay video once; adaptive sampling into a database of 64x48 grey thumbnails "
            "with exact timestamps (dense where the picture changes, around cuts/flashes; >=1 frame per max_gap s; "
            "duplicates capped)",
            "3/4 exact (brute-force BLAS) coarse retrieval of every reference frame against every database frame",
            "5 detailed multi-scale comparison of diversified candidates; best ones re-decoded at 128x96 and rescored",
            "6 Viterbi optimisation of the whole sequence (similarity vs. rhythm/continuity, bounded accuracy loss)",
            "7 refinement: second-chance search for the worst frames + dense decoding of every frame around the "
            "most promising candidates; repeated until the gain is negligible",
            "export: native frames at exact timestamps -> H.264 at the reference frame rate + reference audio",
        ],
        "composition_measures": "multi-scale tone layout (L1 at 24x32,12x16,6x8,3x4), light/dark centroids, spread, "
                                "left-right / top-bottom balance",
        "structural_measures": "SSIM (48x64, 96x128), zero-mean normalised correlation, balanced figure/ground IoU, "
                               "silhouette-contour chamfer similarity (distance transform), gradient-orientation "
                               "histograms",
        "semantic_measures": matcher.sem_status,
        "weights": {"composition": w["composition"], "structural": w["structural"], "semantic": w["semantic"]},
        "coarse_retrieval": "1-RMSE tone (12x16, 24x32), ZNCC (24x32), soft Dice of light and dark regions (24x32)",
        "temporal": "Viterbi; pair cost = lambda_rhythm*|output change - reference change| + lambda_jump*scene jump",
    }


def limitations(matcher) -> list[str]:
    out = [
        "Similarity scores are algorithm-specific and do not represent human-perceived similarity percentages.",
        "The gameplay footage is not a silhouette animation: even the best frames only approximate Bad Apple!!'s "
        "pure black/white shapes; frames are shown unmodified, so colours, HUD and text remain visible.",
        "Adaptive sampling stores a subset of frames; skipped frames are near-duplicates of stored ones and frames "
        "around the most promising candidates are decoded densely in refinement, but a frame that is unique and "
        "far from every promising candidate can still be missed.",
        "Uniform reference frames (e.g. all black) can only be matched by brightness/uniformity.",
        "Dark-world gameplay is predominantly dark, so large white shapes have fewer good candidates than dark ones.",
    ]
    if matcher.sem is None:
        out.append("Semantic similarity was not computed in this run (" + matcher.sem_status + ").")
    return out
