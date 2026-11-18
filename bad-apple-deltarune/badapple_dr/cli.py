"""Command-line interface.

    python -m badapple_dr run --reference badapple.mkv --source deltarune_480p.webm --work work --out output
    python -m badapple_dr gui
    python -m badapple_dr download --out sources
    python -m badapple_dr selftest
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config, set_dotted


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("inputs / outputs")
    g.add_argument("--reference", help="Bad Apple!! reference video file")
    g.add_argument("--reference-audio", help="optional separate audio file for the reference")
    g.add_argument("--source", help="DELTARUNE walkthrough video file used for analysis")
    g.add_argument("--export-source", help="optional higher-quality copy of the SAME walkthrough used for export")
    g.add_argument("--source-info", help="optional yt-dlp .info.json of the walkthrough (chapter names)")
    g.add_argument("--work", help="work folder (caches, checkpoints). Default: work")
    g.add_argument("--out", help="output folder. Default: output")
    g.add_argument("--config", help="JSON config file (see config.example.json)")
    g.add_argument("--ffmpeg", help="path to ffmpeg(.exe)")
    g.add_argument("--ffprobe", help="path to ffprobe(.exe)")
    s = p.add_argument_group("common settings")
    s.add_argument("--ref-start", type=float, help="reference start in seconds (default 0 = literal start)")
    s.add_argument("--duration", type=float, help="seconds of the reference to reconstruct (default 10)")
    s.add_argument("--keep-threshold", type=float, help="adaptive sampling sensitivity (lower = denser database)")
    s.add_argument("--max-gap", type=float, help="keep at least one frame every N seconds")
    s.add_argument("--refine-rounds", type=int, help="targeted refinement rounds (default 2)")
    s.add_argument("--semantic", choices=["auto", "none", "on"], help="semantic (deep feature) similarity")
    s.add_argument("--semantic-model", help="e.g. facebook/dinov2-small or openai/clip-vit-base-patch32")
    s.add_argument("--device", help="auto | cpu | cuda (semantic model)")
    s.add_argument("--hwaccel", help="ffmpeg hardware decoding for the big pass: none | auto | d3d11va | cuda")
    s.add_argument("--workers", type=int, help="parallel ffmpeg processes for random frame access (default 4)")
    s.add_argument("--output-size", help="'source' (default) or WIDTHxHEIGHT (uniform resize + letterbox)")
    s.add_argument("--crf", type=int, help="x264 quality of the final video (lower = better, default 14)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set weights.semantic=0.2 (repeatable)")
    s.add_argument("--verbose", action="store_true")


def _build_cfg(a) -> dict:
    cfg = load_config(a.config)
    m = {"reference": "reference", "reference_audio": "reference_audio", "source": "source",
         "export_source": "export_source", "source_info": "source_info", "work": "work_dir", "out": "out_dir",
         "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "ref_start": "ref_start", "duration": "ref_duration"}
    for k, ck in m.items():
        v = getattr(a, k, None)
        if v is not None:
            cfg[ck] = v
    dotted = {"keep_threshold": "index.keep_threshold", "max_gap": "index.max_gap",
              "refine_rounds": "refine.rounds", "semantic": "semantic.mode", "semantic_model": "semantic.model",
              "device": "semantic.device", "hwaccel": "index.hwaccel", "workers": "rerank.workers",
              "output_size": "export.output_size", "crf": "export.crf"}
    for k, path in dotted.items():
        v = getattr(a, k, None)
        if v is not None:
            set_dotted(cfg, path, v)
    for kv in a.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv}")
        k, v = kv.split("=", 1)
        set_dotted(cfg, k.strip(), v.strip())
    return cfg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="badapple_dr",
                                 description="Recreate Bad Apple!! from unmodified DELTARUNE gameplay frames.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run (or resume) the whole pipeline")
    _add_common(pr)
    pr.add_argument("--until", choices=["probe", "index", "reference", "match", "export"], default="export",
                    help="stop after this stage")
    pr.add_argument("--stop-after-frames", type=int, help=argparse.SUPPRESS)
    sub.add_parser("gui", help="simple graphical interface")
    pd = sub.add_parser("download", help="download the two source videos with yt-dlp")
    pd.add_argument("--out", default="sources")
    pd.add_argument("--quality", default="480", help="analysis copy height (default 480)")
    pd.add_argument("--with-export-copy", action="store_true", help="also download a 720p copy for a sharper output")
    pd.add_argument("--only", choices=["reference", "source", "export"], help="download only one file")
    pd.add_argument("--dry-run", action="store_true", help="show formats and sizes, download nothing")
    pd.add_argument("--cookies-from-browser", help="passed to yt-dlp (only if YouTube asks you to sign in)")
    pt = sub.add_parser("selftest", help="validate the software on synthetic test videos (never real output)")
    pt.add_argument("--work", default="selftest_work")
    pt.add_argument("--semantic-test", action="store_true",
                    help="also exercise the semantic code path with a randomly initialised model (needs torch)")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .pipeline import run

        cfg = _build_cfg(a)
        try:
            run(cfg, until=a.until, stop_after_frames=a.stop_after_frames, verbose=a.verbose)
        except KeyboardInterrupt:
            print("\nInterrupted. Progress is saved; run the same command again to resume.", file=sys.stderr)
            sys.exit(130)
    elif a.cmd == "gui":
        from .gui import main as gui_main

        gui_main()
    elif a.cmd == "download":
        from .download import main as dl_main

        sys.exit(dl_main(a))
    elif a.cmd == "selftest":
        from .selftest import main as st_main

        sys.exit(st_main(a.work, semantic_test=a.semantic_test))


if __name__ == "__main__":
    main()
