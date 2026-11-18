"""Software self-test on SYNTHETIC videos (never used as source imagery for a real reconstruction).

It generates
  * a synthetic silhouette "reference" (24 fps, 4:3, 12 s) with known segments and an audio track
    with beeps at known times;
  * a synthetic pixel-art "gameplay library" (30 fps, 16:9 with a pillarboxed 4:3 game area, VP9
    WebM like YouTube) made of overworld / battle / dialogue / fade / flash scenes, PLUS
      - planted segments: gameplay-styled renderings of the reference shapes (textured, shifted,
        rescaled -- never identical), with known timestamps (ground truth);
      - decoys: uniform frames with the same mean brightness, mirrored layouts, inverted polarity;
  * a 1280x720 H.264 copy of the same library (to test exporting from a second rendition);
then runs the complete pipeline and checks every requirement that can be verified automatically.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .ffmpeg import _NO_WINDOW, find_ffmpeg, find_ffprobe, probe
from .util import LOG, atomic_write_json, setup_logging

REF_FPS, REF_W, REF_H, REF_DUR = 24, 480, 360, 12.0
SRC_FPS, SRC_W, SRC_H = 30, 854, 480
GAME_W, GAME_H, IW, IH = 640, 480, 160, 120
BEEPS = [0.5, 2.0, 4.0, 6.0, 8.0, 9.5, 11.0]
SEGMENTS = [("black", 0.0, 1.0), ("disk", 1.0, 3.0), ("figure", 3.0, 5.0), ("inverse", 5.0, 7.0),
            ("blobs", 7.0, 8.5), ("band", 8.5, 10.0), ("beyond", 10.0, 12.0)]


# ---------------------------------------------------------------------------------------------
# reference fixture
# ---------------------------------------------------------------------------------------------
def _figure(img, cx, cy, s, color):
    cv2.circle(img, (int(cx), int(cy - 1.1 * s)), int(0.42 * s), color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), int(cy + 0.4 * s)), (int(0.5 * s), int(0.95 * s)), 0, 0, 360, color, -1, cv2.LINE_AA)
    pts = np.array([[cx - 0.55 * s, cy - 1.2 * s], [cx - 1.1 * s, cy + 0.2 * s], [cx - 0.3 * s, cy - 0.7 * s]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.line(img, (int(cx + 0.4 * s), int(cy)), (int(cx + 1.0 * s), int(cy - 0.5 * s)), color, max(2, int(0.18 * s)), cv2.LINE_AA)


def ref_frame(i: int) -> np.ndarray:
    t = i / REF_FPS
    S = 2
    W, H = REF_W * S, REF_H * S
    img = np.zeros((H, W), np.uint8)
    label = next(l for l, a, b in SEGMENTS if a <= t < b)
    if label == "disk":
        u = (t - 1.0) / 2.0
        cv2.circle(img, (W // 2, H // 2), int(H * (0.10 + 0.25 * u)), 255, -1, cv2.LINE_AA)
    elif label == "figure":
        u = (t - 3.0) / 2.0
        _figure(img, W * (0.2 + 0.6 * u), H * 0.55, H * 0.22, 255)
    elif label == "inverse":
        u = (t - 5.0) / 2.0
        img[:] = 255
        _figure(img, W * (0.8 - 0.6 * u), H * 0.5, H * 0.26, 0)
    elif label == "blobs":
        u = (t - 7.0) / 1.5
        cv2.ellipse(img, (int(W * 0.3), int(H * 0.45)), (int(W * 0.17), int(H * (0.28 + 0.05 * u))), 15, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.circle(img, (int(W * 0.75), int(H * (0.35 + 0.1 * u))), int(H * 0.09), 255, -1, cv2.LINE_AA)
        cv2.rectangle(img, (0, int(H * 0.86)), (W, H), 255, -1)
    elif label == "band":
        u = (t - 8.5) / 1.5
        img[:] = 255
        ang = math.radians(20 + 50 * u)
        c, s = math.cos(ang), math.sin(ang)
        L = 2 * W
        p0 = np.array([W / 2 - c * L, H / 2 - s * L])
        p1 = np.array([W / 2 + c * L, H / 2 + s * L])
        cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), 0, int(H * 0.22), cv2.LINE_AA)
    elif label == "beyond":
        yy, xx = np.mgrid[0:H, 0:W]
        img[((xx // 80 + yy // 80) % 2) == 0] = 255
    return cv2.resize(img, (REF_W, REF_H), interpolation=cv2.INTER_AREA)


def ref_label(i: int) -> str:
    t = i / REF_FPS
    return next(l for l, a, b in SEGMENTS if a <= t < b)


def write_wav(path: Path, dur: float, rate: int = 48000):
    t = np.arange(int(dur * rate)) / rate
    sig = 0.05 * np.sin(2 * np.pi * 220 * t)
    for b in BEEPS:
        m = (t >= b) & (t < b + 0.06)
        sig[m] += 0.5 * np.sin(2 * np.pi * 1000 * (t[m] - b))
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(st.tobytes())


def make_reference(ffmpeg: str, path: Path) -> None:
    wav = path.with_suffix(".wav")
    write_wav(wav, REF_DUR)
    n = int(REF_DUR * REF_FPS)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s",
           f"{REF_W}x{REF_H}", "-framerate", str(REF_FPS), "-i", "pipe:0", "-i", str(wav), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=_NO_WINDOW)
    for i in range(n):
        p.stdin.write(ref_frame(i).tobytes())  # type: ignore[union-attr]
    p.stdin.close()  # type: ignore[union-attr]
    if p.wait() != 0:
        raise RuntimeError("could not encode the reference fixture")
    wav.unlink()


# ---------------------------------------------------------------------------------------------
# gameplay-like library fixture
# ---------------------------------------------------------------------------------------------
class Lib:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.dark_pal = self.rng.integers(18, 85, (12, 3))
        self.light_pal = self.rng.integers(120, 215, (12, 3))

    def tilemap(self, tw, th, pal):
        ids = self.rng.integers(0, len(pal), (th, tw))
        tiles = pal[ids]  # (th, tw, 3)
        img = np.repeat(np.repeat(tiles, 8, 0), 8, 1).astype(np.int16)
        img += self.rng.integers(-6, 7, img.shape)  # tile texture
        img[::8, :, :] = img[::8, :, :] * 0.7
        img[:, ::8, :] = img[:, ::8, :] * 0.7
        return np.clip(img, 0, 255).astype(np.uint8)

    def sprite(self, img, x, y, color, w=8, h=12):
        x, y = int(x), int(y)
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 10, 10), -1)
        cv2.rectangle(img, (x + 1, y + 1), (x + w - 1, y + h - 1), tuple(int(c) for c in color), -1)
        cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + 5), (225, 205, 190), -1)

    def overworld(self, frames, light=False):
        pal = self.light_pal if light else self.dark_pal
        bg = self.tilemap(40, 15, pal)
        col = [self.rng.integers(60, 255, 3) for _ in range(3)]
        x0 = self.rng.uniform(10, 120)
        for k in range(frames):
            off = (k // 2) % (bg.shape[1] - IW)
            f = bg[:, off:off + IW].copy()
            for j, c in enumerate(col):
                self.sprite(f, (x0 + 25 * j + 0.5 * k) % 150, 60 + 10 * math.sin(k / 9 + j), c)
            yield f

    def battle(self, frames):
        f0 = np.zeros((IH, IW, 3), np.uint8)
        g = tuple(int(c) for c in self.rng.integers(20, 70, 3))
        for x in range(0, IW, 12):
            cv2.line(f0, (x, 0), (x, IH), g, 1)
        for y in range(0, IH, 12):
            cv2.line(f0, (0, y), (IW, y), g, 1)
        party = [self.rng.integers(80, 255, 3) for _ in range(3)]
        enemy = tuple(int(c) for c in self.rng.integers(60, 255, 3))
        board = self.rng.random() < 0.6
        bul = self.rng.uniform(0, 1, (12, 2))
        for k in range(frames):
            f = f0.copy()
            for j, c in enumerate(party):
                self.sprite(f, 12, 20 + 28 * j, c, 10, 16)
            ey = 30 + 6 * math.sin(k / 7)
            cv2.ellipse(f, (120, int(ey) + 15), (18, 22), 0, 0, 360, enemy, -1)
            cv2.rectangle(f, (0, 104), (IW, IH), (0, 0, 0), -1)
            cv2.rectangle(f, (2, 106), (IW - 3, IH - 2), (60, 200, 255), 1)
            if board:
                cv2.rectangle(f, (55, 45), (105, 85), (255, 255, 255), 1)
                for b in bul:
                    bx = 56 + ((b[0] * 48 + 0.7 * k) % 48)
                    by = 46 + ((b[1] * 38 + 0.4 * k) % 38)
                    f[int(by):int(by) + 2, int(bx):int(bx) + 2] = 255
            yield f

    def dialogue(self, frames):
        bg = self.tilemap(20, 15, self.dark_pal)
        chars = [(8 + 6 * (j % 24), 88 + 9 * (j // 24)) for j in range(60)]
        for k in range(frames):
            f = bg.copy()
            cv2.rectangle(f, (4, 82), (IW - 5, IH - 4), (0, 0, 0), -1)
            cv2.rectangle(f, (4, 82), (IW - 5, IH - 4), (255, 255, 255), 1)
            for j, (x, y) in enumerate(chars[: min(len(chars), k // 2)]):
                f[y:y + 5, x:x + 3] = 235
            yield f

    def black(self, frames):
        for _ in range(frames):
            yield np.zeros((IH, IW, 3), np.uint8)

    def flash(self, frames):
        for _ in range(frames):
            f = np.full((IH, IW, 3), 235, np.uint8)
            f = np.clip(f.astype(np.int16) + self.rng.integers(-15, 16, f.shape), 0, 255).astype(np.uint8)
            yield f

    def textured(self, mask, light_base, k, seed):
        r = np.random.default_rng(seed + k // 3)
        light = np.clip(np.array(light_base)[None, None, :] + r.integers(-22, 23, (IH, IW, 3)), 0, 255)
        dark = self.tilemap(20, 15, self.dark_pal) if k == 0 or not hasattr(self, "_dark") else self._dark
        self._dark = dark
        f = np.where(mask[..., None], light, dark).astype(np.uint8)
        self.sprite(f, 140 - (k % 40), 100, (40, 180, 250), 6, 9)  # a small 'gameplay' element
        return f


def internal_mask(ref_gray: np.ndarray, dx: float, dy: float, scale: float, mirror=False, invert=False) -> np.ndarray:
    m = cv2.resize(ref_gray, (IW, IH), interpolation=cv2.INTER_AREA).astype(np.float32)
    M = cv2.getRotationMatrix2D((IW / 2, IH / 2), 0, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    border = 255 if m[0, 0] > 127 else 0
    m = cv2.warpAffine(m, M, (IW, IH), flags=cv2.INTER_LINEAR, borderValue=border)
    if mirror:
        m = m[:, ::-1]
    b = m >= 128
    return ~b if invert else b


def make_library(ffmpeg: str, path: Path, hq_path: Path, seed=7, random_seconds=200.0):
    lib = Lib(seed)
    rng = np.random.default_rng(seed + 1)
    planted_specs = [("disk", (225, 235, 245), 3, -2, 0.95), ("figure", (200, 240, 250), -4, 2, 1.04),
                     ("inverse", (215, 230, 240), 2, 3, 0.97), ("blobs", (230, 225, 250), -3, -2, 1.03),
                     ("band", (220, 240, 235), 4, 1, 0.96)]
    scenes = []
    t_acc = 0.0
    kinds = ["overworld", "overworld_light", "battle", "battle", "dialogue", "overworld"]
    while t_acc < random_seconds:
        k = kinds[int(rng.integers(0, len(kinds)))]
        sec = float(rng.uniform(3, 9))
        scenes.append(("random", k, sec))
        t_acc += sec
        r = rng.random()
        if r < 0.15:
            scenes.append(("random", "black", 1.0))
        elif r < 0.25:
            scenes.append(("random", "flash", 0.15))
    specials = [("planted",) + s for s in planted_specs]
    specials += [("decoy", "uniform", 0), ("decoy", "mirror", "figure"), ("decoy", "mirror", "blobs"),
                 ("decoy", "invert", "disk")]
    for s in specials:
        scenes.insert(int(rng.integers(2, len(scenes) - 1)), s)
    ref_cache = {i: ref_frame(i) for i in range(int(10 * REF_FPS))}
    seg = {l: (a, b) for l, a, b in SEGMENTS}
    truth = {"planted": [], "decoys": []}
    enc = []
    for out, (w, h), vargs in ((path, (SRC_W, SRC_H), ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
                                                        "-row-mt", "1", "-b:v", "1500k"]),
                               (hq_path, (1280, 720), ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"])):
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
               f"{SRC_W}x{SRC_H}", "-framerate", str(SRC_FPS), "-i", "pipe:0"]
        if (w, h) != (SRC_W, SRC_H):
            cmd += ["-vf", f"scale={w}:{h}:flags=lanczos"]
        cmd += vargs + ["-pix_fmt", "yuv420p", str(out)]
        enc.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=_NO_WINDOW))
    x0 = (SRC_W - GAME_W) // 2
    canvas = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    nframe = 0

    def emit(f):
        nonlocal nframe
        canvas[:, x0:x0 + GAME_W] = cv2.resize(f, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
        b = canvas.tobytes()
        for p in enc:
            p.stdin.write(b)  # type: ignore[union-attr]
        nframe += 1

    for sc in scenes:
        start = nframe / SRC_FPS
        if sc[0] == "random":
            _, k, sec = sc
            nf = max(1, int(round(sec * SRC_FPS)))
            gen = {"overworld": lambda: lib.overworld(nf), "overworld_light": lambda: lib.overworld(nf, True),
                   "battle": lambda: lib.battle(nf), "dialogue": lambda: lib.dialogue(nf),
                   "black": lambda: lib.black(nf), "flash": lambda: lib.flash(nf)}[k]()
            for f in gen:
                emit(f)
        elif sc[0] == "planted":
            _, label, color, dx, dy, s = sc
            a, b = seg[label]
            nf = int(round((b - a) * SRC_FPS))
            for k in range(nf):
                i = int(a * REF_FPS + math.floor(k * REF_FPS / SRC_FPS))
                emit(lib.textured(internal_mask(ref_cache[i], dx, dy, s), color, k, 100 + k))
            truth["planted"].append({"label": label, "start": start, "end": nframe / SRC_FPS})
        else:
            _, kind, arg = sc
            if kind == "uniform":
                # brightness-only trap: uniform frames with the SAME mean brightness as every reference frame
                for i in range(int(1 * REF_FPS), int(10 * REF_FPS), 2):
                    v = float(ref_cache[i].mean())
                    emit(np.full((IH, IW, 3), int(round(v)), np.uint8))
            else:
                a, b = seg[arg]
                nf = int(round((b - a) * SRC_FPS))
                for k in range(nf):
                    i = int(a * REF_FPS + math.floor(k * REF_FPS / SRC_FPS))
                    m = internal_mask(ref_cache[i], 0, 0, 1.0, mirror=(kind == "mirror"), invert=(kind == "invert"))
                    emit(lib.textured(m, (225, 235, 245), k, 300 + k))
            truth["decoys"].append({"kind": kind, "of": arg, "start": start, "end": nframe / SRC_FPS})
    for p in enc:
        p.stdin.close()  # type: ignore[union-attr]
        if p.wait() != 0:
            raise RuntimeError("could not encode the library fixture")
    truth["frames"] = nframe
    truth["duration"] = nframe / SRC_FPS
    return truth


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
def beep_onsets(ffmpeg: str, path: str, rate=8000) -> list[float]:
    r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate),
                        "-f", "f32le", "pipe:1"], capture_output=True, creationflags=_NO_WINDOW)
    x = np.frombuffer(r.stdout, np.float32)
    win = int(0.005 * rate)
    e = np.convolve(x * x, np.ones(win) / win, mode="same")
    on = e > 0.02
    idx = np.where(on[1:] & ~on[:-1])[0] + 1
    out, last = [], -1.0
    for i in idx:
        t = i / rate
        if t - last > 0.2:
            out.append(t)
            last = t
    return out


def _csv(path: Path) -> list[dict]:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baselines(work: Path, truth: dict) -> dict:
    """Accuracy of simple baselines on the same database (for comparison with the full matcher)."""
    from .index import SourceDB
    from .reference import load_reference
    from . import features as F

    db = SourceDB(work / "db").open()
    ref = load_reference(work / "reference")
    th = F.to_float(np.asarray(db.thumbs))
    t16 = F.pool2(F.pool2(th)).reshape(len(th), -1)
    means = th.reshape(len(th), -1).mean(1)
    rm = F.to_float(ref["mid"])
    r16 = F.pool2(F.pool2(rm)).reshape(len(rm), -1)
    pts = db.meta["pts"]
    res = {}
    for name in ("brightness_only", "tone16_l2"):
        hits = tot = 0
        for i in range(len(rm)):
            lab = ref_label(i)
            pl = [p for p in truth["planted"] if p["label"] == lab]
            if not pl:
                continue
            if name == "brightness_only":
                j = int(np.argmin(np.abs(means - rm[i].mean())))
            else:
                j = int(np.argmin(((t16 - r16[i]) ** 2).sum(1)))
            tot += 1
            hits += int(pl[0]["start"] - 0.05 <= pts[j] <= pl[0]["end"] + 0.05)
        res[name] = hits / max(tot, 1)
    db.close()
    return res


def main(work_dir: str = "selftest_work", semantic_test: bool = False) -> int:
    work = Path(work_dir).resolve()
    fx = work / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    setup_logging(work)
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe(None, ffmpeg)
    t0 = time.time()
    ref_path, lib_path, hq_path = fx / "synthetic_reference.mkv", fx / "synthetic_library.webm", fx / "synthetic_library_720p.mp4"
    truth_path = fx / "truth.json"
    if not (ref_path.exists() and lib_path.exists() and hq_path.exists() and truth_path.exists()):
        LOG.info("Generating synthetic fixtures (software validation only)...")
        make_reference(ffmpeg, ref_path)
        truth = make_library(ffmpeg, lib_path, hq_path)
        truth_path.write_text(json.dumps(truth, indent=1))
    truth = json.loads(truth_path.read_text())
    LOG.info("Fixtures ready (%.1fs): library %.1f s / %d frames.", time.time() - t0, truth["duration"], truth["frames"])

    from .pipeline import run

    def cfg_for(name: str, export_source=None, sem=False):
        cfg = load_config(None, {
            "reference": str(ref_path), "source": str(lib_path), "export_source": export_source or "",
            "work_dir": str(work / name), "out_dir": str(work / name / "output"),
            "retrieval": {"coarse_topk": 3000, "candidates": 300},
            "rerank": {"fine_topn": 24},
            "semantic": {"mode": "on" if sem else "none"},
        })
        if sem:
            cfg["semantic"]["_random_init_for_tests"] = True
        return cfg

    checks: dict = {}
    timings: dict = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        LOG.info("[%s] %s %s", "PASS" if ok else "FAIL", name, detail)

    # 1) full run
    t = time.time()
    res = run(cfg_for("run_main"))
    timings["full_pipeline_s"] = time.time() - t
    out = work / "run_main" / "output"
    rep = json.loads((out / "quality_report.json").read_text())
    rows = _csv(out / "frame_mapping.csv")
    final = out / "Bad_Apple_Deltarune_10s.mp4"
    info = probe(str(final), ffprobe=ffprobe, ffmpeg=ffmpeg)
    check("output files exist", all((out / n).exists() for n in ("Bad_Apple_Deltarune_10s.mp4", "frame_mapping.csv",
                                                                   "comparison_10s.mp4", "quality_report.json")))
    check("240 reference frames (10 s @ 24 fps)", rep["number_of_reference_frames"] == 240, str(rep["number_of_reference_frames"]))
    check("CSV has one row per output frame", len(rows) == 240, str(len(rows)))
    v = rep["verification"]
    check("output decodes to exactly 240 frames", v.get("decoded_output_frames") == 240, str(v.get("decoded_output_frames")))
    check("output duration is 10.000 s", abs((info.get("duration") or 0) - 10.0) < 0.02, f"{info.get('duration')}")
    check("output frame rate equals reference (24/1)", info["video"]["fps"] == "24/1", info["video"]["fps"])
    check("audio track present, 10 s", bool(info["audio"]) and abs((v.get("audio_duration_s") or 0) - 10.0) < 0.05,
          f"{v.get('audio_duration_s')}")
    check("audio offset vs reference within 2 ms", v.get("audio_offset_ms") is not None and abs(v["audio_offset_ms"]) <= 2.0,
          f"{v.get('audio_offset_ms')} ms")
    on = beep_onsets(ffmpeg, str(final))
    exp_beeps = [b for b in BEEPS if b < 10.0]
    near = all(any(abs(o - b) < 0.012 for o in on) for b in exp_beeps)
    check("beeps at the right times (sync) and none after 10 s", near and not any(o > 10.0 for o in on),
          f"onsets={[round(o, 3) for o in on]}")
    check("every output frame equals its selected source frame",
          v.get("frames_identical_to_selected_source_frames") == 240 and (v.get("per_frame_mad_max") or 99) < 4.0,
          f"identical={v.get('frames_identical_to_selected_source_frames')} max MAD={v.get('per_frame_mad_max')}")
    tm = max(float(r["trace_mad"]) for r in rows)
    check("timestamps re-decode to the analysed frames exactly", tm == 0.0, f"max trace MAD {tm}")
    # accuracy against ground truth
    hits = tot = 0
    decoy_hits = 0
    black_ok = True
    per_label: dict = {}
    for r in rows:
        i = int(r["output_frame_index"])
        lab = ref_label(i)
        ts = float(r["source_timestamp_s"])
        if lab == "black":
            black_ok &= float(r["tone_layout"]) > 0.9
            continue
        pl = [p for p in truth["planted"] if p["label"] == lab][0]
        ok = pl["start"] - 0.05 <= ts <= pl["end"] + 0.05
        hits += ok
        tot += 1
        per_label.setdefault(lab, [0, 0])
        per_label[lab][0] += ok
        per_label[lab][1] += 1
        decoy_hits += any(d["start"] - 0.05 <= ts <= d["end"] + 0.05 for d in truth["decoys"])
    acc = hits / max(tot, 1)
    check("selects the planted matching segment for >= 90% of frames", acc >= 0.90,
          f"{acc:.3f} per segment={ {k: f'{a}/{b}' for k, (a, b) in per_label.items()} }")
    check("never selects a decoy (same brightness / mirrored / inverted)", decoy_hits == 0, f"{decoy_hits} decoy frames")
    check("black reference frames matched by dark frames", black_ok)
    base = baselines(work / "run_main", truth)
    check("brightness-only baseline is clearly worse (sanity of the test)", base["brightness_only"] < acc - 0.3,
          f"brightness-only {base['brightness_only']:.3f}, tone16-L2 {base['tone16_l2']:.3f}, full {acc:.3f}")
    # 2) resume test: interrupted index must equal an uninterrupted one
    t = time.time()
    rcfg = cfg_for("run_resume")
    run(rcfg, until="index", stop_after_frames=2345)
    run(rcfg, until="index", stop_after_frames=5100)
    run(rcfg, until="index")
    timings["resume_test_s"] = time.time() - t
    from .index import SourceDB

    a = SourceDB(work / "run_main" / "db").open()
    b = SourceDB(work / "run_resume" / "db").open()
    same = (len(a.meta) == len(b.meta) and np.array_equal(a.meta["n"], b.meta["n"])
            and np.allclose(a.meta["pts"], b.meta["pts"]) and np.array_equal(np.asarray(a.thumbs), np.asarray(b.thumbs)))
    sa, sb = a.state(), b.state()
    check("resumed index == uninterrupted index", same and sa["decoded"] == sb["decoded"] == truth["frames"],
          f"rows {len(a.meta)} vs {len(b.meta)}, decoded {sa['decoded']} vs {sb['decoded']} (truth {truth['frames']})")
    a.close()
    b.close()
    # 3) second run of the same command must reuse everything
    t = time.time()
    run(cfg_for("run_main"))
    dt = time.time() - t
    timings["rerun_cached_s"] = dt
    check("re-running reuses checkpoints", dt < max(60.0, 0.5 * timings["full_pipeline_s"]), f"{dt:.1f}s")
    # 4) export from a second rendition (720p H.264 copy)
    t = time.time()
    shutil.rmtree(work / "run_hq", ignore_errors=True)
    shutil.copytree(work / "run_main", work / "run_hq", ignore=shutil.ignore_patterns("output", "probe.json"))
    run(cfg_for("run_hq", export_source=str(hq_path)))
    timings["export_rendition_s"] = time.time() - t
    hq_out = work / "run_hq" / "output"
    hrows = _csv(hq_out / "frame_mapping.csv")
    hinfo = probe(str(hq_out / "Bad_Apple_Deltarune_10s.mp4"), ffprobe=ffprobe, ffmpeg=ffmpeg)
    mx = max(float(r["trace_mad"]) for r in hrows)
    check("export from 720p rendition: size + frame match", hinfo["video"]["width"] == 1280 and mx < 6.0,
          f"{hinfo['video']['width']}x{hinfo['video']['height']}, max cross-rendition MAD {mx:.2f}")
    # 5) optional semantic code path (random weights, plumbing only)
    if semantic_test:
        from .semantic import availability

        ok, why = availability()
        if ok:
            shutil.rmtree(work / "run_sem", ignore_errors=True)
            shutil.copytree(work / "run_main", work / "run_sem", ignore=shutil.ignore_patterns("output", "match"))
            run(cfg_for("run_sem", sem=True))
            srows = _csv(work / "run_sem" / "output" / "frame_mapping.csv")
            vals = [r["semantic_similarity"] for r in srows]
            check("semantic code path runs (random weights)", all(v != "" for v in vals), f"{len(vals)} values")
        else:
            check("semantic code path runs (random weights)", False, why)
    summary = {"checks": checks, "timings": timings, "accuracy_full": acc, "baselines": base,
               "truth": {"library_frames": truth["frames"], "planted": truth["planted"], "decoys": truth["decoys"]},
               "database": rep["source_frames"], "mean_similarity": rep["similarity"]["mean_total"]}
    atomic_write_json(work / "selftest_report.json", summary)
    n_fail = sum(1 for c in checks.values() if not c["pass"])
    LOG.info("Self-test finished: %d checks, %d failed. Report: %s", len(checks), n_fail, work / "selftest_report.json")
    return 1 if n_fail else 0
