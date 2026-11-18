"""Detailed similarity scoring between reference frames (targets) and gameplay candidates.

Score families (each in [0, 1], higher = more similar):

composition  -- WHERE light and dark are: multi-scale tone layout (24x32 ... 3x4) and global
                balance/placement (light & dark centroids, spread, left/right & top/bottom balance).
structural   -- SHAPE of the light/dark regions: SSIM, zero-mean correlation, figure/ground
                overlap (balanced IoU), silhouette-contour chamfer similarity, gradient orientations;
                at the mid level (48x64) and, when available, the fine level (96x128).
semantic     -- optional deep-feature similarity (see semantic.py), added by the matcher.

For (almost) uniform reference frames (e.g. an all-black frame) structure is undefined; the
structural score then measures how uniform the candidate is.
"""
from __future__ import annotations

import numpy as np

from . import features as F

STRUCT_KEYS = ("ssim_mid", "zncc", "biou_mid", "chamfer_mid", "hog", "ssim_fine", "biou_fine", "chamfer_fine")


class TargetSet:
    """Precomputed features for every reference frame (fine: (T, Hf, Wf) uint8)."""

    def __init__(self, fine_u8: np.ndarray, weights: dict):
        self.w = weights
        self.fine = F.to_float(fine_u8)
        self.mid = F.pool2(self.fine)
        self.T = self.fine.shape[0]
        self.levels = F.pyramid(self.mid, 5)  # list of (T, h, w)
        self.std = self.mid.reshape(self.T, -1).std(axis=1)
        self.uniform = self.std < float(weights.get("uniform_std", 0.04))
        self.mask_mid = self.mid > 0.5
        self.mask_fine = self.fine > 0.5
        self.edge_mid = F.contour(self.mask_mid)
        self.edge_fine = F.contour(self.mask_fine)
        self.dist_mid = F.dist_batch(self.edge_mid)
        self.dist_fine = F.dist_batch(self.edge_fine)
        self.ss_mid = F.ssim_stats(self.mid)
        self.ss_fine = F.ssim_stats(self.fine)
        self.hog_h, self.hog_e = F.hog(self.mid)
        self.bal = F.balance(self.mid)
        self.l1 = self.levels[1]

    def g16(self) -> np.ndarray:
        """12x16 tone thumbnails (for temporal change measurements)."""
        return self.levels[2]


class CandBatch:
    """Features of a batch of gameplay candidates.

    mid_u8: (B, H, W) uint8 (database thumbnails, or pooled fine frames); fine_u8: optional (B, 2H, 2W).
    """

    def __init__(self, mid_u8: np.ndarray, fine_u8: np.ndarray | None = None):
        self.B = mid_u8.shape[0]
        self.mid = F.to_float(mid_u8)
        self.levels = F.pyramid(self.mid, 5)
        self.std = self.mid.reshape(self.B, -1).std(axis=1)
        self.thr = F.source_thresholds(self.mid)
        self.mask_mid = self.mid > self.thr[:, None, None]
        self.edge_mid = F.contour(self.mask_mid)
        self.dist_mid = F.dist_batch(self.edge_mid)
        self.ss_mid = F.ssim_stats(self.mid)
        self.hog_h, self.hog_e = F.hog(self.mid)
        self.bal = F.balance(self.mid)
        self.has_fine = fine_u8 is not None
        if self.has_fine:
            self.fine = F.to_float(fine_u8)
            self.mask_fine = self.fine > self.thr[:, None, None]
            self.edge_fine = F.contour(self.mask_fine)
            self.dist_fine = F.dist_batch(self.edge_fine)
            self.ss_fine = F.ssim_stats(self.fine)

    def subset(self, idx) -> "CandBatch":
        """Candidates `idx` of this batch, reusing the already computed per-frame features."""
        idx = np.asarray(idx, np.int64)
        o = CandBatch.__new__(CandBatch)
        o.B = len(idx)
        for k in ("mid", "std", "thr", "mask_mid", "edge_mid", "dist_mid", "hog_h", "hog_e", "bal"):
            setattr(o, k, getattr(self, k)[idx])
        o.levels = [lv[idx] for lv in self.levels]
        o.ss_mid = (self.ss_mid[0][idx], self.ss_mid[1][idx])
        o.has_fine = self.has_fine
        if self.has_fine:
            for k in ("fine", "mask_fine", "edge_fine", "dist_fine"):
                setattr(o, k, getattr(self, k)[idx])
            o.ss_fine = (self.ss_fine[0][idx], self.ss_fine[1][idx])
        return o


def score(ts: TargetSet, i: int, cb: CandBatch) -> dict:
    """Score candidates of `cb` against target i. Returns dict of (B,) float32 arrays."""
    w = ts.w
    B = cb.B
    out: dict = {}
    # ---------------- composition ----------------
    alphas = w["tone_levels"]
    tone = np.zeros(B, np.float32)
    for k, a in enumerate(alphas, start=1):
        d = np.abs(ts.levels[k][i][None] - cb.levels[k]).mean(axis=(-2, -1))
        tone += a * (1.0 - d)
    tone /= max(sum(alphas), 1e-9)
    bal = F.balance_sim(ts.bal[i], cb.bal)
    cp = w["composition_parts"]
    comp = (cp["tone"] * tone + cp["balance"] * bal) / (cp["tone"] + cp["balance"])
    out["tone"] = tone
    out["balance"] = bal
    out["composition"] = comp.astype(np.float32)
    # ---------------- structure ----------------
    sp = w["structural_parts"]
    if ts.uniform[i]:
        struct = np.clip(1.0 - cb.std / 0.12, 0.0, 1.0).astype(np.float32)
        for k in STRUCT_KEYS:
            out[k] = np.full(B, np.nan, np.float32)
        out["uniform_target"] = np.ones(B, bool)
    else:
        a = ts.mid[i]
        parts = {}
        parts["ssim_mid"] = np.clip(F.ssim(a, ts.ss_mid[0][i], ts.ss_mid[1][i], cb.mid, *cb.ss_mid), 0, 1)
        z1 = F.zncc(a, cb.mid)
        z2 = F.zncc(ts.l1[i], cb.levels[1])
        parts["zncc"] = np.clip(0.5 * (z1 + z2), 0, 1)
        parts["biou_mid"] = F.balanced_iou(ts.mask_mid[i], cb.mask_mid)
        tau_m = 0.1 * a.shape[1]
        parts["chamfer_mid"] = F.chamfer_sim(ts.edge_mid[i], ts.dist_mid[i], cb.edge_mid, cb.dist_mid, tau_m)
        parts["hog"] = np.clip(F.hog_sim(ts.hog_h[i], ts.hog_e[i], cb.hog_h, cb.hog_e), 0, 1)
        if cb.has_fine:
            af = ts.fine[i]
            parts["ssim_fine"] = np.clip(F.ssim(af, ts.ss_fine[0][i], ts.ss_fine[1][i], cb.fine, *cb.ss_fine), 0, 1)
            parts["biou_fine"] = F.balanced_iou(ts.mask_fine[i], cb.mask_fine)
            parts["chamfer_fine"] = F.chamfer_sim(ts.edge_fine[i], ts.dist_fine[i], cb.edge_fine, cb.dist_fine,
                                                  0.1 * af.shape[1])
        num = np.zeros(B, np.float32)
        den = 0.0
        for k, v in parts.items():
            num += sp[k] * v
            den += sp[k]
        struct = num / max(den, 1e-9)
        for k in STRUCT_KEYS:
            out[k] = parts.get(k, np.full(B, np.nan, np.float32)).astype(np.float32)
        out["uniform_target"] = np.zeros(B, bool)
    out["structural"] = struct.astype(np.float32)
    out["semantic"] = np.full(B, np.nan, np.float32)
    out["total"] = combine_total(out["composition"], out["structural"], None, w)
    return out


def combine_total(comp, struct, sem, w: dict) -> np.ndarray:
    wc, ws, wm = float(w["composition"]), float(w["structural"]), float(w["semantic"])
    if sem is None or wm <= 0:
        return ((wc * comp + ws * struct) / (wc + ws)).astype(np.float32)
    sem = np.asarray(sem, np.float32)
    ok = np.isfinite(sem)
    tot = np.where(ok, (wc * comp + ws * struct + wm * np.where(ok, sem, 0)) / (wc + ws + wm),
                   (wc * comp + ws * struct) / (wc + ws))
    return tot.astype(np.float32)
