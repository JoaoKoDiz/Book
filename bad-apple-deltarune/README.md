# Bad Apple!! × DELTARUNE — frame-matching reconstruction

Recreates the first 10 seconds of the Bad Apple!! shadow-art video using **only real, unmodified
frames** of the DELTARUNE Chapters 1–5 walkthrough. For every reference frame the program searches the
**entire** 18-hour walkthrough for the most similar gameplay frame (composition and structure, not
just brightness), optimises the whole sequence for coherence, and exports a playable MP4 with the
reference audio, a frame-by-frame mapping CSV, a side-by-side diagnostic video and a quality report.

| | |
|---|---|
| Source library | <https://www.youtube.com/watch?v=egkMsxfREXs> — "DELTARUNE - Full Game Walkthrough (Chapter 1,2,3,4,5)", 65 134 s (18 h 05 min) |
| Reference | <https://www.youtube.com/watch?v=9lNZ_Rnr7Jc> — "Bad Apple!! - Full Version w/video [Lyrics in Romaji, Translation in English]", 324 s, 24 fps, 4:3 |
| Output | `Bad_Apple_Deltarune_10s.mp4`, `frame_mapping.csv`, `comparison_10s.mp4`, `quality_report.json`, `contact_sheet.png` |

> The durations and frame rates above are from YouTube's metadata. The program re-measures them from the
> downloaded files (stage 1) and uses only the measured values. YouTube lists the reference at
> **24 fps**, so 10 s = exactly **240 frames**. If your downloaded file reports something else, the
> program adapts automatically.

**What it never does:** no AI image/video generation, no diffusion/inpainting/interpolation, no
recolouring, no black-and-white conversion, no thresholding/inversion/posterisation of output frames,
no cropping/rotating/flipping/stretching/compositing, no overlays or transitions, no external images.
Grey conversion, cropping to the game area, thresholds and edge maps are used **internally for
similarity measurement only**. Every output frame is a frame decoded at an exact timestamp of the
walkthrough and passed to the encoder in its native size and pixel format. The program then decodes
the finished MP4 again and checks every frame against the source frame it claims to be.

---

## 1. Quick start (Windows 11)

Open **Terminal** (PowerShell) in this folder and run:

```bash
winget install --id Python.Python.3.12 -e
```

```bash
winget install --id Gyan.FFmpeg -e
```

```bash
winget install DenoLand.Deno
```

Close the terminal and open a **new** one, so the new programs are on `PATH`. Then:

```bash
setup_windows.bat
```

```bash
.venv\Scripts\activate
```

```bash
python -m badapple_dr selftest
```

```bash
python -m badapple_dr download --out sources
```

```bash
run_windows.bat
```

The results land in `output\`. You can stop the program at any time (Ctrl+C, closing the window, or a
crash). Running the same command again resumes where it stopped.

Prefer clicking? After `setup_windows.bat` and the download, run `python -m badapple_dr gui`, select
the two videos, and press **Start / Resume**.

### Measured performance (development machine: 4 CPU cores, no GPU)

| Stage | Measurement | Extrapolation to the 18 h walkthrough |
|---|---|---|
| 2 (one pass over the video) | 932 frames/s, = 90 % of raw FFmpeg decode speed (1047 frames/s) on a synthetic 480p VP9 stream | ~1.95 M frames → **~20–45 min** |
| 4 (exact retrieval) | 240 targets × 600,000 database frames in **72 s**, 0.54 GB RAM | ~1–2 min |
| 5a (detailed comparison) | 177 ms per target (400 candidates) | ~45 s |
| 5b–7 + export | 2.1 min on the self-test | ~10–30 min (more candidates, longer seeks) |

### What each step does

| Step | Time (typical PC) | Disk |
|---|---|---|
| `setup_windows.bat` | 2–5 min | ~0.4 GB (`.venv`) |
| `selftest` (optional, recommended) | ~5–10 min | ~0.5 GB in `selftest_work\` (delete afterwards) |
| `download` | depends on your internet | ~1.9 GB (480p VP9 copy) + ~25 MB reference; +3.4 GB with `--with-export-copy` |
| `run` | **~1–3 h** total; mostly the one-time pass over the 18 h video | ~1.5–3 GB in `work\` |

---

## 2. Requirements

* **Windows 10/11 64-bit** (Linux and macOS work the same way, using `python3` and paths with `/`).
* **Python 3.10–3.13**.
* **FFmpeg**. `winget install --id Gyan.FFmpeg -e` provides `ffmpeg` + `ffprobe`. If FFmpeg is not
  installed, the bundled binary from the `imageio-ffmpeg` package is used automatically. Without
  `ffprobe`, file inspection falls back to parsing `ffmpeg -i`, which works but is less detailed.
* **RAM:** 4 GB is enough. Data are streamed and memory-mapped. Measured: exact retrieval over a
  600,000-frame database peaks at **0.54 GB** of real memory; the mapped database pages are
  reclaimable OS cache.
* **Disk:** ~2 GB for the analysis copy, ~2–3 GB for the work folder, plus ~3.4 GB for the optional
  720p export copy.
* **CPU / GPU:** the core pipeline is CPU-only (NumPy/OpenCV + FFmpeg) and runs on any x86-64 CPU. A GPU
  is **not** needed. Optional speed-ups:
  * `--hwaccel auto` (or `d3d11va`) lets FFmpeg decode the big pass on the GPU (Intel/AMD/NVIDIA). Try it
    if the first stage runs below ~800 frames/s.
  * The optional semantic stage (`requirements-semantic.txt`) uses PyTorch. It runs on CPU, or on an
    NVIDIA GPU with CUDA if you install the CUDA build of torch (`--device cuda`). It processes only
    the final candidates (a few thousand frames), so CPU is fine.

---

## 3. Getting the source videos

`python -m badapple_dr download --out sources` uses yt-dlp to download:

* `sources\badapple_reference.mkv`: Bad Apple!! upload `9lNZ_Rnr7Jc`. This is the best video ≤720p
  that is not one of YouTube's AI "super-resolution" (`-sr`) upscales, merged with its audio.
* `sources\deltarune_480p.webm` (+ `.info.json` with the YouTube chapters): the walkthrough at 480p,
  30 fps, video only, ~1.9 GB. The game renders at 640×480, so 480p already holds the full game picture.
* with `--with-export-copy`: `sources\deltarune_720p30.webm` (~3.4 GB). This is the same video at
  720p, used only to export sharper frames. The program checks that every exported frame is the
  analysed one.

Options: `--dry-run` shows formats and sizes without downloading, `--quality 360` gives a smaller
analysis copy, and `--only reference|source|export` downloads one file.

**Supplying files yourself:** any file of the same videos works. Pass them with `--reference` and
`--source` (any container/codec FFmpeg can read). The reference must be the `9lNZ_Rnr7Jc` upload: the
program uses the literal first 10 seconds of whatever file you give it.

---

## 4. Running

```bash
python -m badapple_dr run --reference sources\badapple_reference.mkv --source sources\deltarune_480p.webm --export-source sources\deltarune_720p30.webm --work work --out output
```

`--export-source` is optional; without it, frames are exported from the analysis copy.

Useful options (all of them: `python -m badapple_dr run --help`):

| Option | Meaning |
|---|---|
| `--until probe\|index\|reference\|match` | stop after a stage (e.g. check `work\probe.json` first) |
| `--keep-threshold 0.006` | denser source database (default 0.010; lower = more frames, more disk) |
| `--refine-rounds 3` | more targeted-refinement rounds (default 2) |
| `--semantic none\|auto\|on` | deep-feature similarity (default `auto` = used if torch/transformers are installed) |
| `--hwaccel auto` | GPU video decoding for the big pass |
| `--workers 6` | parallel FFmpeg processes for random frame access (default 4) |
| `--output-size 1920x1080` | uniform resize + letterbox of the final video (default: native size) |
| `--ref-start 2.5` | use a different reference start. **Default 0 = the literal start.** |
| `--set weights.structural=0.5` | change any value of `config.example.json` |
| `--config my_config.json` | load a JSON configuration |

The literal first 10 seconds are used by default (`--ref-start 0`). Look at
`work\reference\reference_contact_sheet.png` after the `reference` stage to see exactly what those
frames contain (this upload may begin with a title card or intro).

---

## 5. How it works

```
 stage 1  inspect files (ffprobe): duration, fps, resolution, aspect, codecs, audio;
          detect the 4:3 game area inside black bars (analysis only, frames sampled over the whole file)
 stage 2  ONE sequential decode of the whole 18 h video (FFmpeg -> pipe, exact timestamps per frame)
          adaptive sampling -> database of 64x48 grey thumbnails (append-only, checkpointed, resumable):
            keep when the picture changed >= keep_threshold since the last kept frame
              (static scenes sparse; walking / battles / cutscenes dense, automatically)
            keep every frame near cuts, flashes and fast transitions
            keep >= 1 frame per max_gap seconds; identical-looking frames stored at most dup_limit times
 stage 3  descriptors computed from the database in blocks (no re-decoding of the video)
 stage 4  EXACT coarse retrieval: every reference frame vs every database frame as BLAS matrix
          products (tone layout 12x16/24x32, correlation, light+dark Dice) -> top 6000 / target
          -> temporal diversification -> 400 candidates per target
 stage 5  detailed multi-scale comparison (mid 48x64); best 32 per target RE-DECODED at 96x128 and
          rescored (+ optional DINOv2/CLIP semantic score)
 stage 6  Viterbi over the whole sequence: similarity vs. rhythm & continuity, with a hard cap on how
          much similarity a frame may lose for coherence (max_drop)
 stage 7  refinement rounds: wider second-chance search for the worst 25 % of frames; dense decoding
          of EVERY frame around the most promising candidates (recovers frames the sampler skipped);
          re-optimise; repeat until the gain is negligible
 export   native frames at exact timestamps -> H.264 at the reference fps + the reference audio;
          decode the result again and verify frames, duration and audio sync
```

### Similarity measures (all computed on internal analysis copies, never on output frames)

* **Composition** (weight 0.50). This is where light and dark are: L1 tone layout at 24×32, 12×16, 6×8
  and 3×4 (coarse levels weighted for overall composition), light *and* dark centroids, spread,
  left/right and top/bottom balance. A mostly white gameplay screen cannot win just because its
  average brightness is right; the uniform-frame decoys in the self-test check exactly this.
* **Structure** (weight 0.40). This is the *shape* of the light and dark regions: SSIM (48×64, 96×128),
  zero-mean normalised correlation, balanced figure/ground IoU (light and dark regions must both
  overlap), silhouette-contour chamfer similarity via distance transforms (48×64, 96×128), and
  gradient-orientation histograms.
* **Semantic** (weight 0.10, optional). This uses DINOv2 (default `facebook/dinov2-small`) or CLIP
  image embeddings: global similarity plus a 3×4 patch-layout similarity ("what kind of content is
  where"). It is secondary to composition, as requested. The model only *measures* similarity.
* **Temporal coherence**: the change between consecutive output frames should follow the change
  between consecutive reference frames (no flicker on static shots, movement when the reference
  moves), and jumps to unrelated parts of the video are discouraged unless the reference itself cuts.

FAISS was evaluated and not used. With only ~240 query frames, exact brute-force search via BLAS over
the memory-mapped database takes seconds to minutes and has perfect recall, whereas an approximate
index would lose candidates for no speed benefit.

---

## 6. Output files

* **`Bad_Apple_Deltarune_10s.mp4`**: H.264 (CRF 14, yuv420p) at the reference frame rate (24 fps → 240
  frames = 10.000 s), native resolution of the export source, with the reference's first 10 s of
  audio (AAC 192 kbps).
* **`frame_mapping.csv`**: one row per output frame, with these columns: output and reference frame
  index and timestamp; **source frame index**, **source timestamp** (seconds and `HH:MM:SS.mmm`) and
  chapter; total, composition, structural, semantic (empty if not computed) and temporal-coherence
  scores; the individual measures (tone, balance, SSIM, ZNCC, IoU, chamfer, orientation); how the
  frame was found; the export timestamp; and two verification values:
  * `trace_mad`: re-decoding the exported timestamp through the analysis path reproduces the analysed
    image; 0 means identical.
  * `output_mad`: the final MP4's frame vs the analysed image; small values come from H.264
    compression only.
* **`comparison_10s.mp4`**: diagnostic only. The reference (left) next to the selected gameplay frame
  (right), with a text strip showing timestamps and scores.
* **`quality_report.json`**: frame counts, number of source frames analysed/stored/re-decoded,
  unique frames used, average/median/minimum similarity, the 15 worst frames, processing time per
  stage, sampling density (per minute and per chapter), methodology, configuration, verification
  results and limitations.
* **`contact_sheet.png`**: reference/selection pairs for a quick visual check.

**Similarity scores are specific to this algorithm.** They are weighted combinations of the measures
above, scaled to 0–1. They are **not** percentages of human-perceived similarity.

---

## 7. Resuming, checkpoints and disk usage

* `work\db\`: the source database (`thumbs.u8` ~3 KB per stored frame, `meta.bin`, `state.json`). It
  is checkpointed every 6000 frames / 30 s. After an interruption, the files are truncated to the last
  checkpoint and decoding continues right after the last committed frame; frame numbers stay exact
  (the self-test checks this).
* `work\fine_cache\`: re-decoded candidate frames (12 KB each).
* `work\match\`: retrieval results, candidate pools after each refinement round, statistics.
* `work\probe.json`, `work\reference\`, `work\timings.json`, `work\pipeline.log` (full log).

Changing settings later only recomputes what depends on them. Matching weights reuse the database;
index settings require a new database (use a new `--work` folder, or delete `work\db`). The work
folder can be deleted once you are happy with the result.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `FFmpeg was not found` | `winget install --id Gyan.FFmpeg -e`, then open a **new** terminal; or `pip install imageio-ffmpeg`; or `--ffmpeg C:\path\ffmpeg.exe` |
| yt-dlp: *"Sign in to confirm you're not a bot"* | usually only on VPN/shared networks: try another network or later. yt-dlp's documented option is `--cookies-from-browser firefox` (also accepted by our `download` command); see the yt-dlp FAQ on cookies |
| yt-dlp: *"No supported JavaScript runtime"* / missing formats | `winget install DenoLand.Deno`, new terminal; `pip install -U "yt-dlp[default]"` |
| yt-dlp: *"Requested format is not available"* | run `python -m badapple_dr download --dry-run`; update yt-dlp; `--quality 360` |
| Stage 2 is slow (< 500 frames/s) | try `--hwaccel auto`; close other heavy programs; use the 360p or 480p analysis copy (never 1080p for analysis) |
| `database ... built from a different source file or settings` | you changed the source or index settings: use a new `--work` folder or delete `work\db` |
| Disk full | delete `selftest_work\`; use `--quality 360`; skip `--with-export-copy`; raise `--keep-threshold` to 0.015 |
| Out of memory | lower `--set retrieval.chunk=4096` and `--workers 2` |
| `torch`/`transformers` errors | semantic features are optional: `--semantic none`; or reinstall with `pip install -r requirements-semantic.txt` |
| Hugging Face download fails | first run needs internet for the model (~90 MB); or `--semantic none` |
| Output looks too dark/mismatched in places | see `quality_report.json → worst_matching_frames`; try `--keep-threshold 0.006 --refine-rounds 3` |
| `work\probe.json` shows a wrong "analysis area" (e.g. a decorative border was kept) | set it by hand, analysis only: `--set "analysis.source_crop=[107,0,640,480]"` (x, y, width, height in source pixels), then use a new `--work` folder |
| The reference starts with an intro/title | that is the literal start, as requested. Only if you decide otherwise: `--ref-start <seconds>` |

---

## 9. Validation (`selftest`)

`python -m badapple_dr selftest` generates **synthetic** test videos. These are a silhouette reference
with known segments and beeps at known times, and a pixel-art "gameplay" library with planted
matching segments and decoys (same-brightness uniform frames, mirrored and inverted layouts). It runs
the complete pipeline and checks:

* loading, probing and game-area detection;
* frame extraction and exact timestamps;
* the resumed database is identical to an uninterrupted one;
* retrieval accuracy against ground truth, never picking a decoy, and comparison with the
  brightness-only baseline;
* exact traceability: timestamps re-decode to the analysed frames;
* every output frame equals its selected source frame;
* frame count and duration (exactly 10.000 s at 24 fps);
* audio presence, length and sync (cross-correlation and beep onsets);
* export from a second rendition;
* reuse of checkpoints.

Synthetic fixtures validate the software only and are never used as imagery for the real reconstruction.

---

## 10. Project layout

```
badapple_dr/
  cli.py        command line            pipeline.py  stage orchestration / resume
  ffmpeg.py     decoding with exact timestamps, probing, random frame access
  probe.py      stage 1, game-area detection
  index.py      stage 2/3 source database (adaptive sampling, checkpoints)
  reference.py  reference window extraction
  features.py   image measures (SSIM, ZNCC, IoU, chamfer, HOG, balance, Otsu, ...)
  scoring.py    composition / structural scores
  retrieval.py  stage 4 exact coarse retrieval + diversification
  match.py      stages 5-7 (detailed comparison, refinement rounds)
  temporal.py   stage 6 Viterbi
  semantic.py   optional DINOv2/CLIP similarity
  export.py     MP4, CSV, comparison video, report, verification
  download.py   yt-dlp helper          gui.py  Tkinter front-end
  selftest.py   synthetic validation suite
tests/          pytest wrappers (unit tests + self-test)
```
