"""FFmpeg / ffprobe helpers.

* locating the binaries (PATH, explicit path, or the copy bundled with `imageio-ffmpeg`)
* probing files (ffprobe JSON, with a fallback parser for `ffmpeg -i` when ffprobe is missing)
* `FrameReader`: streams raw frames from ffmpeg together with the EXACT presentation timestamp of
  every frame (parsed from the `showinfo` filter), so that each analysed frame can be traced back to
  a real timestamp in the source video.
* `fetch_frames`: random access to specific frames by timestamp (clustered, parallel).

All decoding uses `-copyts`, so every timestamp is the raw timestamp stored in the file.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

from .util import LOG

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console popups on Windows


class FFmpegError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# locating binaries
# ----------------------------------------------------------------------------------------------
def _exists(c: str | None) -> str | None:
    if not c:
        return None
    if Path(c).is_file():
        return str(Path(c))
    w = shutil.which(c)
    return w


def find_ffmpeg(explicit: str | None = None) -> str:
    for c in (explicit, os.environ.get("BADAPPLE_FFMPEG"), "ffmpeg"):
        found = _exists(c)
        if found:
            return found
    try:
        import imageio_ffmpeg  # bundled static ffmpeg (Windows/macOS/Linux wheels)

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    raise FFmpegError(
        "FFmpeg was not found. Install it (Windows: `winget install --id Gyan.FFmpeg -e`, then open a "
        "new terminal), or `pip install imageio-ffmpeg`, or pass --ffmpeg C:\\path\\to\\ffmpeg.exe")


def find_ffprobe(explicit: str | None = None, ffmpeg_path: str | None = None) -> str | None:
    for c in (explicit, os.environ.get("BADAPPLE_FFPROBE"), "ffprobe"):
        found = _exists(c)
        if found:
            return found
    if ffmpeg_path:
        sib = Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if sib.is_file():
            return str(sib)
    return None


_version_cache: dict[str, tuple] = {}


def ffmpeg_version(ffmpeg: str) -> tuple:
    if ffmpeg in _version_cache:
        return _version_cache[ffmpeg]
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True,
                           errors="replace", creationflags=_NO_WINDOW, timeout=30)
        m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", r.stdout)
        v = (int(m.group(1)), int(m.group(2))) if m else (99, 0)
    except Exception:
        v = (99, 0)
    _version_cache[ffmpeg] = v
    return v


def passthrough_args(ffmpeg: str) -> list[str]:
    """1 decoded frame -> 1 output frame (no duplication / dropping)."""
    return ["-fps_mode", "passthrough"] if ffmpeg_version(ffmpeg) >= (5, 1) else ["-vsync", "passthrough"]


def run_ffmpeg(cmd: list[str], what: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    LOG.debug("run: %s", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW, timeout=timeout)
    if r.returncode != 0:
        tail = r.stderr.decode("utf-8", "replace").strip().splitlines()[-15:]
        raise FFmpegError(f"{what} failed (exit {r.returncode}):\n" + "\n".join(tail))
    return r


# ----------------------------------------------------------------------------------------------
# probing
# ----------------------------------------------------------------------------------------------
def _fr(s) -> Fraction | None:
    try:
        if s in (None, "", "0/0", "N/A"):
            return None
        f = Fraction(str(s))
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _flt(s, default=None):
    try:
        if s in (None, "", "N/A"):
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def probe(path: str, ffprobe: str | None = None, ffmpeg: str | None = None) -> dict:
    """Return a normalised description of a media file (duration, video/audio stream properties)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)]
        r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
        if r.returncode != 0:
            raise FFmpegError(f"ffprobe could not read {path}: " + r.stderr.decode("utf-8", "replace")[-800:])
        data = json.loads(r.stdout.decode("utf-8", "replace"))
        info = _normalise_ffprobe(data)
    elif ffmpeg:
        info = _probe_with_ffmpeg(ffmpeg, str(p))
    else:
        raise FFmpegError("Neither ffprobe nor ffmpeg is available for probing.")
    info["path"] = str(p.resolve())
    info["size_bytes"] = p.stat().st_size
    if info.get("video") is None:
        raise FFmpegError(f"No video stream found in {path}")
    return info


def _video_entry(s: dict) -> dict:
    r_fps = _fr(s.get("r_frame_rate"))
    a_fps = _fr(s.get("avg_frame_rate"))
    fps = r_fps
    if fps is None or fps > 240:
        fps = a_fps
    if fps is None:
        fps = Fraction(30, 1)
    cfr = bool(r_fps and a_fps and abs(float(r_fps) - float(a_fps)) / float(r_fps) < 0.01)
    nb = s.get("nb_frames")
    tags = s.get("tags") or {}
    dur = _flt(s.get("duration"))
    if dur is None:  # matroska/webm keep per-stream duration in tags ("DURATION": "00:01:02.345000000")
        t = tags.get("DURATION") or tags.get("duration")
        if t:
            try:
                hh, mm, ss = t.split(":")
                dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
            except ValueError:
                dur = None
    return {
        "index": s.get("index"),
        "codec": s.get("codec_name"),
        "profile": s.get("profile"),
        "width": int(s.get("width") or 0),
        "height": int(s.get("height") or 0),
        "sar": s.get("sample_aspect_ratio") or "1:1",
        "dar": s.get("display_aspect_ratio"),
        "pix_fmt": s.get("pix_fmt"),
        "fps": f"{fps.numerator}/{fps.denominator}",
        "fps_float": float(fps),
        "r_frame_rate": s.get("r_frame_rate"),
        "avg_frame_rate": s.get("avg_frame_rate"),
        "cfr_likely": cfr,
        "time_base": s.get("time_base"),
        "start_time": _flt(s.get("start_time"), 0.0),
        "duration": dur,
        "nb_frames": int(nb) if (nb and str(nb).isdigit()) else None,
        "color_range": s.get("color_range"),
        "color_space": s.get("color_space"),
        "color_transfer": s.get("color_transfer"),
        "color_primaries": s.get("color_primaries"),
        "bit_rate": int(s["bit_rate"]) if str(s.get("bit_rate", "")).isdigit() else None,
    }


def _audio_entry(s: dict) -> dict:
    return {
        "index": s.get("index"),
        "codec": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate") or 0),
        "channels": int(s.get("channels") or 0),
        "channel_layout": s.get("channel_layout"),
        "start_time": _flt(s.get("start_time"), 0.0),
        "duration": _flt(s.get("duration")),
        "bit_rate": int(s["bit_rate"]) if str(s.get("bit_rate", "")).isdigit() else None,
    }


def _normalise_ffprobe(data: dict) -> dict:
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    info = {
        "format_name": fmt.get("format_name"),
        "duration": _flt(fmt.get("duration")),
        "start_time": _flt(fmt.get("start_time"), 0.0),
        "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        "video": _video_entry(vids[0]) if vids else None,
        "video_streams": len(vids),
        "audio": [_audio_entry(a) for a in auds],
        "probe_tool": "ffprobe",
    }
    if info["duration"] is None and info["video"] and info["video"]["duration"]:
        info["duration"] = info["video"]["duration"]
    return info


def _probe_with_ffmpeg(ffmpeg: str, path: str) -> dict:
    """Fallback when ffprobe is missing: parse the stream summary printed by `ffmpeg -i`."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, creationflags=_NO_WINDOW)
    txt = r.stderr.decode("utf-8", "replace")
    info = {"format_name": None, "duration": None, "start_time": 0.0, "bit_rate": None,
            "video": None, "video_streams": 0, "audio": [], "probe_tool": "ffmpeg -i (ffprobe missing)"}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?),\s*start:\s*(-?[\d.]+)", txt)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        info["start_time"] = float(m.group(4))
    m = re.search(r"Input #0, ([^,]+)", txt)
    if m:
        info["format_name"] = m.group(1)
    for line in txt.splitlines():
        sm = re.search(r"Stream #0:(\d+)[^:]*: (Video|Audio): (.*)", line)
        if not sm:
            continue
        idx, kind, rest = int(sm.group(1)), sm.group(2), sm.group(3)
        if kind == "Video" and info["video"] is None and "attached pic" not in rest:
            codec = rest.split()[0].strip(",")
            pm = re.search(r",\s*([a-z0-9_]+)(?:\(([^)]*)\))?,\s*(\d{2,5})x(\d{2,5})", rest)
            fm = re.search(r"([\d.]+(?:/[\d.]+)?)\s*fps", rest)
            tm = re.search(r"([\d.]+)k?\s*tbr", rest)
            fps = Fraction(fm.group(1)).limit_denominator(1001) if fm else (
                Fraction(tm.group(1)).limit_denominator(1001) if tm else Fraction(30))
            v = {
                "index": idx, "codec": codec, "profile": None,
                "width": int(pm.group(3)) if pm else 0, "height": int(pm.group(4)) if pm else 0,
                "sar": "1:1", "dar": None, "pix_fmt": pm.group(1) if pm else None,
                "fps": f"{fps.numerator}/{fps.denominator}", "fps_float": float(fps),
                "r_frame_rate": None, "avg_frame_rate": None, "cfr_likely": True, "time_base": None,
                "start_time": info["start_time"], "duration": info["duration"], "nb_frames": None,
                "color_range": None, "color_space": None, "color_transfer": None, "color_primaries": None,
                "bit_rate": None,
            }
            if pm and pm.group(2):
                bits = [b.strip() for b in pm.group(2).split(",")]
                if bits and bits[0] in ("tv", "pc"):
                    v["color_range"] = bits[0]
                if len(bits) > 1:
                    v["color_space"] = bits[1].split("/")[0]
            sarm = re.search(r"SAR (\d+:\d+) DAR (\d+:\d+)", rest)
            if sarm:
                v["sar"], v["dar"] = sarm.group(1), sarm.group(2)
            info["video"] = v
            info["video_streams"] += 1
        elif kind == "Audio":
            am = re.search(r"(\d+) Hz", rest)
            info["audio"].append({
                "index": idx, "codec": rest.split()[0].strip(","),
                "sample_rate": int(am.group(1)) if am else 0,
                "channels": 2 if "stereo" in rest else (1 if "mono" in rest else 0),
                "channel_layout": None, "start_time": info["start_time"], "duration": info["duration"],
                "bit_rate": None,
            })
    return info


# ----------------------------------------------------------------------------------------------
# streaming frames with exact timestamps
# ----------------------------------------------------------------------------------------------
_SHOWINFO = re.compile(rb"\] n:\s*(\d+)\s+pts:\s*(\S+)\s+pts_time:\s*(\S+)")
_TIMEBASE = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")

_BPP = {"gray": 1.0, "rgb24": 3.0, "bgr24": 3.0, "yuv420p": 1.5}


class FrameReader:
    """Iterate over frames of a video as numpy arrays, each with its exact timestamp.

    Parameters
    ----------
    crop : (x, y, w, h) region used for ANALYSIS (e.g. the game area inside pillarbox bars).
    size : (W, H) output size; the (cropped) frame is scaled with area averaging.
    fit  : (W, H) scale preserving aspect ratio into W x H and pad with black (reference analysis).
    seek : seconds RELATIVE TO THE FILE START; decoding starts at the keyframe before and ffmpeg
           discards frames before the position. Callers still filter by timestamp.
    native : output frames at their stored size (no scaling).
    """

    def __init__(self, ffmpeg: str, path: str, *, pix_fmt: str = "gray", size=None, crop=None, fit=None,
                 native_size=None, seek: float | None = None, max_frames: int | None = None,
                 threads: int = 0, hwaccel: str | None = None, extra_input_args=None):
        if pix_fmt not in _BPP:
            raise ValueError(f"unsupported pix_fmt {pix_fmt}")
        vf = []
        if crop is not None:
            x, y, w, h = [int(v) for v in crop]
            vf.append(f"crop={w}:{h}:{x}:{y}")
        if fit is not None:
            W, H = int(fit[0]), int(fit[1])
            vf.append(f"scale={W}:{H}:flags=area:force_original_aspect_ratio=decrease")
            vf.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")
        elif size is not None:
            W, H = int(size[0]), int(size[1])
            vf.append(f"scale={W}:{H}:flags=area")
        else:
            if native_size is None:
                raise ValueError("native_size required when no size/fit is given")
            W, H = int(native_size[0]), int(native_size[1])
            if crop is not None:
                W, H = int(crop[2]), int(crop[3])
        vf.append(f"format={pix_fmt}")
        vf.append("showinfo")
        self.width, self.height, self.pix_fmt = W, H, pix_fmt
        self.shape = (H, W) if pix_fmt == "gray" else ((H, W, 3) if pix_fmt in ("rgb24", "bgr24") else (H * 3 // 2, W))
        self.frame_bytes = int(W * H * _BPP[pix_fmt])
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
        if threads:
            cmd += ["-threads", str(int(threads))]
        if hwaccel and hwaccel != "none":
            cmd += ["-hwaccel", hwaccel]
        cmd += list(extra_input_args or [])
        cmd += ["-copyts"]
        if seek is not None and seek > 0:
            cmd += ["-ss", f"{seek:.6f}"]
        cmd += ["-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", ",".join(vf)]
        cmd += passthrough_args(ffmpeg)
        if max_frames:
            cmd += ["-frames:v", str(int(max_frames))]
        cmd += ["-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
        self.cmd = cmd
        LOG.debug("FrameReader: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW, bufsize=1 << 20)
        self._pts: "queue.Queue[float | None]" = queue.Queue()
        self._tail: deque = deque(maxlen=40)
        self.count = 0
        self._closed = False
        self._t = threading.Thread(target=self._read_stderr, daemon=True)
        self._t.start()

    def _read_stderr(self):
        # Timestamps are computed exactly as integer pts x filter time base (the printed pts_time may
        # be rounded by some ffmpeg builds, which would be ambiguous ~18 hours into a video).
        tb_num = tb_den = None
        try:
            for line in self.proc.stderr:  # type: ignore[union-attr]
                m = _SHOWINFO.search(line)
                if m:
                    try:
                        if tb_num is not None:
                            self._pts.put(int(m.group(2)) * tb_num / tb_den)
                        else:
                            self._pts.put(float(m.group(3)))
                    except ValueError:  # NOPTS
                        self._pts.put(float("nan"))
                    continue
                tm = _TIMEBASE.search(line)
                if tm and tb_num is None:
                    tb_num, tb_den = int(tm.group(1)), int(tm.group(2))
                    if tb_den == 0:
                        tb_num = tb_den = None
                    continue
                self._tail.append(line.decode("utf-8", "replace").rstrip())
        except (ValueError, OSError):
            if not self._closed:  # a read error after close() is expected; anything else is real
                raise
        finally:
            self._pts.put(None)

    def _next_pts(self) -> float:
        while True:
            try:
                v = self._pts.get(timeout=120)
            except queue.Empty:
                raise FFmpegError("timed out waiting for frame timestamps from ffmpeg")
            if v is None:
                raise FFmpegError("ffmpeg delivered a frame without a timestamp:\n" + self.stderr_tail())
            return v

    def stderr_tail(self) -> str:
        return "\n".join(list(self._tail)[-15:])

    def read_batch(self, n: int):
        """Read up to n frames. Returns (pts float64 array, frames uint8 array). Empty at end of stream."""
        buf = bytearray(n * self.frame_bytes)
        mv = memoryview(buf)
        got = 0
        out = self.proc.stdout
        while got < len(buf):
            k = out.readinto(mv[got:])  # type: ignore[union-attr]
            if not k:
                break
            got += k
        nfr = got // self.frame_bytes
        if nfr == 0:
            self._check_exit(empty=True)
            return np.zeros(0, np.float64), np.zeros((0,) + self.shape, np.uint8)
        arr = np.frombuffer(buf, np.uint8, count=nfr * self.frame_bytes).reshape((nfr,) + self.shape)
        pts = np.array([self._next_pts() for _ in range(nfr)], np.float64)
        self.count += nfr
        return pts, arr

    def __iter__(self):
        while True:
            pts, arr = self.read_batch(64)
            if len(pts) == 0:
                return
            for i in range(len(pts)):
                yield float(pts[i]), arr[i]

    def _check_exit(self, empty: bool):
        rc = self.proc.wait()
        if rc != 0 and self.count == 0 and not self._closed:
            raise FFmpegError(f"ffmpeg failed while decoding (exit {rc}):\n{self.stderr_tail()}")

    def close(self):
        self._closed = True
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        # let the timestamp thread drain stderr to EOF before the pipes are closed (no race)
        self._t.join(timeout=10)
        for s in (self.proc.stdout, self.proc.stderr):
            try:
                s.close()  # type: ignore[union-attr]
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_window(ffmpeg: str, path: str, start_pts: float, end_pts: float, *, file_start: float, fps: float,
                max_frames: int | None = None, **reader_kw):
    """Decode every frame with start_pts <= pts <= end_pts (raw timestamps). Returns (pts, frames)."""
    seek = start_pts - file_start - 0.25
    nmax = int((end_pts - start_pts) * fps * 1.2) + 32 + int(0.25 * fps * 2)
    if max_frames:
        nmax = min(nmax, max_frames)
    pts_out, fr_out = [], []
    tol = 1e-4
    with FrameReader(ffmpeg, path, seek=seek if seek > 0 else None, max_frames=nmax, **reader_kw) as rd:
        while True:
            pts, arr = rd.read_batch(64)
            if len(pts) == 0:
                break
            sel = (pts >= start_pts - tol) & (pts <= end_pts + tol)
            if sel.any():
                pts_out.append(pts[sel])
                fr_out.append(arr[sel].copy())
            if pts[-1] > end_pts + tol:
                break
    if not pts_out:
        return np.zeros(0, np.float64), None
    return np.concatenate(pts_out), np.concatenate(fr_out)


def fetch_frames(ffmpeg: str, path: str, pts_list, *, file_start: float, fps: float, workers: int = 4,
                 cluster_gap: float = 1.0, tolerance: float | None = None, progress=None, **reader_kw) -> dict:
    """Fetch specific frames by raw timestamp. Returns {requested_pts: (actual_pts, frame)}.

    Nearby requests are decoded in one pass (clusters), clusters run in parallel ffmpeg processes.
    A request is satisfied only by a frame whose timestamp is within `tolerance` (default half a frame).
    """
    want = sorted(set(float(p) for p in pts_list))
    if not want:
        return {}
    tol = tolerance if tolerance is not None else 0.5 / fps
    clusters, cur = [], [want[0]]
    for p in want[1:]:
        if p - cur[-1] <= cluster_gap and p - cur[0] <= 8.0:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    reader_kw = dict(reader_kw)
    reader_kw.setdefault("threads", 2)

    def work(cl):
        pts, frames = read_window(ffmpeg, path, cl[0] - tol, cl[-1] + tol, file_start=file_start, fps=fps,
                                  **reader_kw)
        res = {}
        if frames is None:
            return res
        for p in cl:
            j = int(np.argmin(np.abs(pts - p)))
            if abs(pts[j] - p) <= tol:
                res[p] = (float(pts[j]), frames[j])
        if progress is not None:
            progress.update(len(cl))
        return res

    out: dict = {}
    if workers <= 1 or len(clusters) == 1:
        for cl in clusters:
            out.update(work(cl))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(work, clusters):
                out.update(res)
    return out


def grab_frames_near(ffmpeg: str, path: str, times, *, file_start: float, workers: int = 4, **reader_kw):
    """First frame at/after each (relative) time; used for coarse sampling (bar detection, previews)."""
    reader_kw = dict(reader_kw)
    reader_kw.setdefault("threads", 2)

    def one(t):
        with FrameReader(ffmpeg, path, seek=max(0.0, float(t)), max_frames=1, **reader_kw) as rd:
            pts, arr = rd.read_batch(1)
        return (float(pts[0]), arr[0].copy()) if len(pts) else None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(one, list(times)))
    return [r for r in res if r is not None]
