"""Stage 4 -- exact coarse retrieval over the WHOLE source database.

With only a few hundred reference frames, an exact brute-force search expressed as matrix products
(BLAS) over the memory-mapped database is fast and has perfect recall with respect to the coarse
score, so no approximate index (FAISS, HNSW...) is needed. The coarse score combines:

  tone16 / tone32  1 - RMSE of the 12x16 / 24x32 tone layouts         (composition)
  zncc32           zero-mean normalised correlation at 24x32         (pattern, brightness-invariant)
  dice32           soft Dice overlap of light AND dark regions, 24x32 (figure/ground)

For uniform reference frames only the tone terms are used (plus a penalty on candidate contrast).
"""
from __future__ import annotations

import numpy as np

from . import features as F
from .util import LOG, Progress


def _desc_from_mid(mid_f: np.ndarray, thr: np.ndarray) -> dict:
    B = mid_f.shape[0]
    l1 = F.pool2(mid_f)            # 24x32
    l2 = F.pool2(l1)               # 12x16
    t32 = l1.reshape(B, -1)
    t16 = l2.reshape(B, -1)
    z = t32 - t32.mean(axis=1, keepdims=True)
    zn = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.where(zn > 1e-6, z / np.maximum(zn, 1e-12), 0.0).astype(np.float32)
    m = F.soft_mask(l1, thr).reshape(B, -1)
    return {"t16": t16, "t32": t32, "z32": z, "m32": m, "std": mid_f.reshape(B, -1).std(axis=1)}


def target_descriptors(ref_mid_u8: np.ndarray) -> dict:
    x = F.to_float(ref_mid_u8)
    return _desc_from_mid(x, np.full(x.shape[0], 0.5, np.float32))


def db_descriptors(mid_u8: np.ndarray) -> dict:
    x = F.to_float(mid_u8)
    return _desc_from_mid(x, F.source_thresholds(x))


def coarse_scores(td: dict, dd: dict, w: dict, uniform: np.ndarray) -> np.ndarray:
    """(T, B) coarse similarity matrix."""
    def rmse(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
        return np.sqrt(np.maximum(d, 0.0) / a.shape[1])

    s16 = 1.0 - rmse(td["t16"], dd["t16"])
    s32 = 1.0 - rmse(td["t32"], dd["t32"])
    z = np.maximum(td["z32"] @ dd["z32"].T, 0.0)
    ma, mb = td["m32"], dd["m32"]
    D = ma.shape[1]
    ab = ma @ mb.T
    sa, sb = ma.sum(1)[:, None], mb.sum(1)[None, :]
    qa, qb = (ma * ma).sum(1)[:, None], (mb * mb).sum(1)[None, :]
    dice_l = np.where(qa + qb > 1e-6, 2 * ab / np.maximum(qa + qb, 1e-6), 1.0)
    # dark regions: (1-a).(1-b) = D - sa - sb + ab ; |1-a|^2 = D - 2 sa + qa
    abd = D - sa - sb + ab
    qad, qbd = D - 2 * sa + qa, D - 2 * sb + qb
    dice_d = np.where(qad + qbd > 1e-6, 2 * abd / np.maximum(qad + qbd, 1e-6), 1.0)
    dice = 0.5 * (dice_l + dice_d)
    cw = w["coarse"]
    S = (cw["tone16"] * s16 + cw["tone32"] * s32 + cw["zncc32"] * z + cw["dice32"] * dice) / (
        cw["tone16"] + cw["tone32"] + cw["zncc32"] + cw["dice32"])
    if uniform.any():
        Su = 0.5 * s16 + 0.5 * s32 - 0.5 * np.maximum(dd["std"][None, :] - 0.05, 0.0)
        S = np.where(uniform[:, None], Su, S)
    return S.astype(np.float32)


def retrieve(db, ref_mid_u8: np.ndarray, uniform: np.ndarray, cfg: dict):
    """Exact top-K coarse candidates (database rows) for every reference frame."""
    rc = cfg["retrieval"]
    w = cfg["weights"]
    K = int(rc["coarse_topk"])
    chunk = int(rc["chunk"])
    thumbs = db.thumbs
    N = thumbs.shape[0]
    T = ref_mid_u8.shape[0]
    K = min(K, N)
    td = target_descriptors(ref_mid_u8)
    best_s = np.full((T, 0), -np.inf, np.float32)
    best_r = np.zeros((T, 0), np.int64)
    prog = Progress(total=N, desc="Coarse retrieval (exact)", unit="rows")
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        dd = db_descriptors(np.asarray(thumbs[a:b]))
        S = coarse_scores(td, dd, w, uniform)
        rows = np.broadcast_to(np.arange(a, b, dtype=np.int64)[None, :], S.shape)
        allS = np.concatenate([best_s, S], axis=1)
        allR = np.concatenate([best_r, rows], axis=1)
        if allS.shape[1] > K:
            idx = np.argpartition(-allS, K - 1, axis=1)[:, :K]
            best_s = np.take_along_axis(allS, idx, axis=1)
            best_r = np.take_along_axis(allR, idx, axis=1)
        else:
            best_s, best_r = allS, np.ascontiguousarray(allR)
        prog.update(b - a)
    prog.close()
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_r, order, 1), np.take_along_axis(best_s, order, 1)


def diversify(rows: np.ndarray, scores: np.ndarray, pts: np.ndarray, limit: int, window: float,
              per_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy temporal diversification: at most `per_window` candidates per `window` seconds,
    so that one long scene cannot fill the whole candidate list."""
    keep_r, keep_s = [], []
    counts: dict = {}
    for r, s in zip(rows, scores):
        if r < 0 or not np.isfinite(s):
            continue
        key = int(pts[r] // window)
        c = counts.get(key, 0)
        if c >= per_window:
            continue
        counts[key] = c + 1
        keep_r.append(int(r))
        keep_s.append(float(s))
        if len(keep_r) >= limit:
            break
    return np.array(keep_r, np.int64), np.array(keep_s, np.float32)
