"""Fast unit tests:  python -m pytest -q tests"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from badapple_dr import features as F
from badapple_dr import scoring as S
from badapple_dr.config import DEFAULTS
from badapple_dr.ffmpeg import FrameReader, fetch_frames, find_ffmpeg, read_window
from badapple_dr.framestore import FrameStore
from badapple_dr.temporal import optimise

H, W = 96, 128


def disk(cx, cy, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) < r * r).astype(np.uint8) * 255


def test_otsu_two_level_midpoint():
    a = np.r_[np.zeros(50), np.ones(50) * 0.8].reshape(10, 10).astype(np.float32)
    assert abs(F.otsu_batch(a[None])[0] - 0.4) < 0.01


def test_ssim_matches_opencv_reference():
    rng = np.random.default_rng(0)
    a = F.to_float(disk(40, 48, 25))
    b = np.clip(a * 0.7 + rng.random(a.shape).astype(np.float32) * 0.3, 0, 1).astype(np.float32)

    def ssim_cv(x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        g = lambda z: cv2.GaussianBlur(z, (7, 7), 1.5, borderType=cv2.BORDER_REFLECT_101)
        mx, my = g(x), g(y)
        vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
        return float((((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))).mean())

    assert abs(F.ssim_pair(a, b) - ssim_cv(a, b)) < 1e-3
    assert abs(F.ssim_pair(a, a) - 1.0) < 1e-3  # float32 rounding of local variances


def test_chamfer_and_iou_identity_and_inversion():
    m = disk(40, 48, 25) > 0
    e = F.contour(m)
    d = F.dist_to(e)
    assert F.chamfer_sim(e, d, e[None], d[None], 12.0)[0] == pytest.approx(1.0)
    assert F.balanced_iou(m, m[None])[0] == pytest.approx(1.0)
    assert F.balanced_iou(m, (~m)[None])[0] < 0.05  # polarity matters


def test_composition_beats_brightness_trap():
    ref = disk(40, 48, 25)
    ts = S.TargetSet(ref[None], DEFAULTS["weights"])
    rng = np.random.default_rng(1)
    textured = np.clip(disk(43, 46, 24) * 0.7 + 40 + rng.integers(0, 40, (H, W)), 0, 255).astype(np.uint8)
    uniform = np.full((H, W), int(ref.mean()), np.uint8)  # identical mean brightness
    mirror = ref[:, ::-1].copy()
    fine = np.stack([textured, uniform, mirror])
    mid = np.round(F.pool2(fine.astype(np.float32))).astype(np.uint8)
    sc = S.score(ts, 0, S.CandBatch(mid, fine))
    assert sc["total"][0] > sc["total"][1] + 0.1
    assert sc["total"][0] > sc["total"][2] + 0.1


def test_uniform_reference_prefers_uniform_candidates():
    ref = np.zeros((H, W), np.uint8)
    ts = S.TargetSet(ref[None], DEFAULTS["weights"])
    assert ts.uniform[0]
    black = np.zeros((H, W), np.uint8)
    noisy = np.random.default_rng(2).integers(0, 60, (H, W)).astype(np.uint8)
    fine = np.stack([black, noisy])
    mid = np.round(F.pool2(fine.astype(np.float32))).astype(np.uint8)
    sc = S.score(ts, 0, S.CandBatch(mid, fine))
    assert sc["total"][0] > sc["total"][1]


def test_viterbi_bounded_accuracy_and_continuity():
    T, K = 6, 3
    rng = np.random.default_rng(3)
    g = rng.random((K, 12, 16)).astype(np.float32)
    cands = []
    for t in range(T):
        # candidate 0 is a continuous sequence, 1/2 are slightly better unrelated frames
        cands.append({"total": np.array([0.80, 0.81, 0.70], np.float32),
                      "n": np.array([100 + t, 5000 + 97 * t, 9000 + 13 * t]),
                      "pts": np.array([(100 + t) / 30, (5000 + 97 * t) / 30, (9000 + 13 * t) / 30]),
                      "g16": g})
    ref = np.repeat(g[:1], T, axis=0)
    path, coh, pc, dref = optimise(cands, ref, DEFAULTS["temporal"])
    assert all(p == 0 for p in path)  # continuity wins within max_drop
    cfg = dict(DEFAULTS["temporal"], max_drop=0.005)
    path2, *_ = optimise(cands, ref, cfg)
    assert all(p == 1 for p in path2)  # accuracy wins when the cap is tight


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    ff = find_ffmpeg()
    p = tmp_path_factory.mktemp("clip") / "c.webm"
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30", "-t", "4",
                    "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-y", str(p)], check=True)
    return ff, str(p)


def test_reader_timestamps_and_random_access(clip):
    ff, p = clip
    with FrameReader(ff, p, pix_fmt="gray", size=(64, 48)) as rd:
        allp, allf = [], []
        while True:
            pts, fr = rd.read_batch(37)
            if len(pts) == 0:
                break
            allp.append(pts)
            allf.append(fr.copy())
    pts = np.concatenate(allp)
    frames = np.concatenate(allf)
    assert len(pts) == 120
    assert np.allclose(np.diff(pts), 1 / 30, atol=0.0015)  # webm: millisecond timestamps
    want = [pts[5], pts[61], pts[118]]
    got = fetch_frames(ff, p, want, file_start=0.0, fps=30.0, workers=2, tolerance=0.004, pix_fmt="gray", size=(64, 48))
    for k, w in enumerate(want):
        i = [5, 61, 118][k]
        assert got[w][0] == pytest.approx(w)
        assert np.array_equal(got[w][1], frames[i])  # exact same pixels as the sequential pass
    wp, wf = read_window(ff, p, pts[30], pts[40], file_start=0.0, fps=30.0, pix_fmt="gray", size=(64, 48))
    assert len(wp) == 11 and np.array_equal(wf, frames[30:41])


def test_framestore_roundtrip(tmp_path):
    fs = FrameStore(tmp_path / "fs", (4, 5))
    a = np.arange(20, dtype=np.uint8).reshape(4, 5)
    fs.add_many([(7, 1.5, a), (9, 2.5, a + 1)])
    fs2 = FrameStore(tmp_path / "fs", (4, 5))  # reopen (resume)
    assert 7 in fs2 and 9 in fs2 and fs2.pts(9) == 2.5
    assert np.array_equal(fs2.get_many([9, 7])[0], a + 1)
