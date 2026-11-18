"""Download the two source videos with yt-dlp (run this on your own computer).

Files written to --out (default "sources"):
  badapple_reference.mkv          Bad Apple!! upload 9lNZ_Rnr7Jc: best NON-upscaled video <=720p + audio
  deltarune_480p.<ext>            DELTARUNE walkthrough egkMsxfREXs, video only, 480p 30 fps (analysis copy)
  deltarune_480p.info.json        metadata incl. YouTube chapters (used for chapter names in the CSV)
  deltarune_720p30.<ext>          optional (--with-export-copy): 720p 30 fps copy used for a sharper output

yt-dlp needs a JavaScript runtime for YouTube (Deno is recommended: `winget install DenoLand.Deno`;
Node.js also works). Partial downloads resume automatically when the command is run again.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .ffmpeg import find_ffmpeg

REF_URL = "https://www.youtube.com/watch?v=9lNZ_Rnr7Jc"
SRC_URL = "https://www.youtube.com/watch?v=egkMsxfREXs"


def _base(ffmpeg: str, cookies: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--ffmpeg-location", ffmpeg, "--retries", "30", "--fragment-retries", "30",
           "--continue", "--no-playlist"]
    if not shutil.which("deno") and shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    return cmd


_VP9 = "[vcodec~='^(vp9|vp09)']"
_AVC = "[vcodec^=avc1]"


def _pref(spec: str) -> str:
    """VP9 first, then H.264 (both decode everywhere and fast), then anything (e.g. AV1)."""
    return f"{spec}{_VP9}/{spec}{_AVC}/{spec}"


def _jobs(out: Path, q: str, with_export: bool):
    ref_v = "bv*[height<=720][format_id!*=sr]"  # never YouTube's AI super-resolution ('-sr') upscales
    jobs = [
        ("reference", REF_URL, "/".join(f"{v}+ba[ext=m4a]/{v}+ba" for v in _pref(ref_v).split("/")) + "/b",
         str(out / "badapple_reference.%(ext)s"), ["--merge-output-format", "mkv"]),
        ("source", SRC_URL, _pref(f"bv*[height={q}][fps<=30]") + f"/bv*[height<={q}][fps<=30]",
         str(out / f"deltarune_{q}p.%(ext)s"), ["--write-info-json"]),
    ]
    if with_export:
        jobs.append(("export", SRC_URL, _pref("bv*[height=720][fps<=30]"), str(out / "deltarune_720p30.%(ext)s"), []))
    return jobs


def main(a) -> int:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("yt-dlp is not installed. Run:  pip install -U \"yt-dlp[default]\"")
        return 2
    if not (shutil.which("deno") or shutil.which("node")):
        print("WARNING: no JavaScript runtime found. YouTube downloads need one. Install Deno:\n"
              "    winget install DenoLand.Deno\n  (then open a NEW terminal) -- or install Node.js.")
    ffmpeg = find_ffmpeg()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = _base(ffmpeg, a.cookies_from_browser)
    jobs = [j for j in _jobs(out, str(a.quality), a.with_export_copy) if not a.only or j[0] == a.only]
    total = 0
    for name, url, fmt, tmpl, extra in jobs:
        r = subprocess.run(base + ["-f", fmt, "--print", "%(format_id)s|%(resolution)s|%(fps)s|%(vcodec)s|"
                                   "%(filesize,filesize_approx)s", "--skip-download", url],
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(f"[{name}] could not query formats:\n{r.stderr[-1500:]}")
            print("If YouTube says 'Sign in to confirm you're not a bot', see the README troubleshooting section.")
            return 1
        line = (r.stdout.strip().splitlines() or ["?"])[-1]
        parts = line.split("|")
        try:
            size = sum(int(float(x)) for x in parts[-1].split("+") if x not in ("NA", "None", ""))
        except ValueError:
            size = 0
        total += size
        print(f"[{name}] format {parts[0]}  {parts[1] if len(parts) > 1 else ''}  ~{size / 1e9:.2f} GB")
    free = shutil.disk_usage(out).free
    print(f"Total download ~{total / 1e9:.2f} GB, free space in {out.resolve()}: {free / 1e9:.1f} GB")
    if a.dry_run:
        return 0
    if total and free < total * 1.15 + 2e9:
        print("Not enough free disk space (need the download size + ~2 GB for the work folder).")
        return 3
    for name, url, fmt, tmpl, extra in jobs:
        print(f"\n=== downloading {name} ===")
        r = subprocess.run(base + ["-f", fmt, "-o", tmpl] + extra + [url])
        if r.returncode != 0:
            print(f"[{name}] download failed (exit {r.returncode}). Re-run the same command to resume.")
            return 1
    files = sorted(p.name for p in out.iterdir() if p.is_file() and not p.name.endswith((".part", ".ytdl")))
    print("\nDownloaded files:", ", ".join(files))
    ref = next((out / f for f in files if f.startswith("badapple_reference.") and not f.endswith(".json")), None)
    src = next((out / f for f in files if f.startswith(f"deltarune_{a.quality}p.") and not f.endswith(".json")), None)
    exp = next((out / f for f in files if f.startswith("deltarune_720p30.") and not f.endswith(".json")), None)
    if ref and src:
        cmd = f'python -m badapple_dr run --reference "{ref}" --source "{src}"'
        if exp:
            cmd += f' --export-source "{exp}"'
        print("\nNext step:\n    " + cmd)
    return 0
