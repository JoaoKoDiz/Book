"""Image measurements used for similarity analysis (never applied to output frames).

Conventions: images are float32 in [0, 1]; single images have shape (H, W), batches (B, H, W).
Functions taking a target `a` and a batch `b` broadcast the target against every batch element.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

EPS = 1e-6


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) * np.float32(1.0 / 255.0)


def pool2(x: np.ndarray) -> np.ndarray:
    """2x2 mean pooling over the last two axes (odd trailing row/column dropped)."""
    h, w = x.shape[-2] // 2, x.shape[-1] // 2
    x = x[..., : 2 * h, : 2 * w].astype(np.float32, copy=False)
    return x.reshape(x.shape[:-2] + (h, 2, w, 2)).mean(axis=(-3, -1), dtype=np.float32)


def pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [x.astype(np.float32, copy=False)]
    for _ in range(levels - 1):
        out.append(pool2(out[-1]))
    return out


# ----------------------------------------------------------------------------------------------
# filtering
# ----------------------------------------------------------------------------------------------
_KCACHE: dict = {}


def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    key = (sigma, radius)
    if key not in _KCACHE:
        t = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-(t ** 2) / (2.0 * sigma * sigma))
        _KCACHE[key] = (k / k.sum()).astype(np.float32)
    return _KCACHE[key]


def blur(x: np.ndarray, sigma: float = 1.5, radius: int = 3) -> np.ndarray:
    """Separable Gaussian blur over the last two axes with reflect padding; any leading batch dims."""
    k = _gauss1d(sigma, radius)
    H, W = x.shape[-2], x.shape[-1]
    lead = [(0, 0)] * (x.ndim - 2)
    xp = np.pad(x, lead + [(radius, radius), (0, 0)], mode="reflect")
    acc = k[0] * xp[..., 0:H, :]
    for i in range(1, len(k)):
        acc += k[i] * xp[..., i:i + H, :]
    xp = np.pad(acc, lead + [(0, 0), (radius, radius)], mode="reflect")
    out = k[0] * xp[..., :, 0:W]
    for i in range(1, len(k)):
        out += k[i] * xp[..., :, i:i + W]
    return out.astype(np.float32, copy=False)


# ----------------------------------------------------------------------------------------------
# SSIM (Wang et al. 2004), Gaussian window, data range 1
# ----------------------------------------------------------------------------------------------
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def ssim_stats(x: np.ndarray):
    mu = blur(x)
    var = blur(x * x) - mu * mu
    return mu, np.maximum(var, 0.0)


def ssim(a, mu_a, var_a, b, mu_b, var_b) -> np.ndarray:
    """Mean SSIM of target a (H,W) against batch b (B,H,W), using precomputed local statistics."""
    cov = blur(a[None] * b) - mu_a[None] * mu_b
    num = (2.0 * mu_a[None] * mu_b + _C1) * (2.0 * cov + _C2)
    den = (mu_a[None] ** 2 + mu_b ** 2 + _C1) * (var_a[None] + var_b + _C2)
    return (num / den).mean(axis=(-2, -1))


def ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
    ma, va = ssim_stats(a)
    mb, vb = ssim_stats(b[None])
    return float(ssim(a, ma, va, b[None], mb, vb)[0])


# ----------------------------------------------------------------------------------------------
# correlation, thresholds, masks
# ----------------------------------------------------------------------------------------------
def zncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation of a (H,W) with each image in b (B,H,W). 0 if flat."""
    a0 = a - a.mean()
    b0 = b - b.mean(axis=(-2, -1), keepdims=True)
    num = (a0[None] * b0).sum(axis=(-2, -1))
    den = np.sqrt(float((a0 * a0).sum()) * (b0 * b0).sum(axis=(-2, -1)))
    return np.where(den > 1e-8, num / np.maximum(den, 1e-12), 0.0).astype(np.float32)


def otsu_batch(x: np.ndarray) -> np.ndarray:
    """Otsu threshold (in [0,1]) of every image of a batch (B,H,W)."""
    B = x.shape[0]
    q = np.clip((x.reshape(B, -1) * 255.0 + 0.5).astype(np.int32), 0, 255)
    idx = q + (np.arange(B, dtype=np.int64) * 256)[:, None]
    hist = np.bincount(idx.ravel(), minlength=256 * B).reshape(B, 256).astype(np.float64)
    p = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    omega = np.cumsum(p, axis=1)
    mu = np.cumsum(p * np.arange(256), axis=1)
    mu_t = mu[:, -1:]
    sb = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    # For images with empty histogram gaps (e.g. two-level images) the maximum is a plateau; take its
    # middle instead of its first index, otherwise the threshold hugs the dark mode.
    smax = sb.max(axis=1, keepdims=True)
    on = sb >= smax * (1.0 - 1e-9)
    first = np.argmax(on, axis=1)
    last = 255 - np.argmax(on[:, ::-1], axis=1)
    k = 0.5 * (first + last)
    return ((k + 0.5) / 255.0).astype(np.float32)


def source_thresholds(mid: np.ndarray) -> np.ndarray:
    """Light/dark split used to read a gameplay frame as figure/ground (analysis only).

    Otsu threshold, clamped to a sensible range; nearly flat frames use the absolute midpoint, so a
    dark frame reads as 'all dark' instead of amplified noise.
    """
    t = np.clip(otsu_batch(mid), 0.10, 0.70)
    std = mid.reshape(mid.shape[0], -1).std(axis=1)
    return np.where(std < 0.02, 0.5, t).astype(np.float32)


def balanced_iou(ma: np.ndarray, mb: np.ndarray) -> np.ndarray:
    """Mean of IoU(light regions) and IoU(dark regions): figure AND ground must overlap."""
    def iou(p, q):
        inter = (p[None] & q).sum(axis=(-2, -1))
        union = (p[None] | q).sum(axis=(-2, -1))
        return np.where(union > 0, inter / np.maximum(union, 1), 1.0)

    return (0.5 * (iou(ma, mb) + iou(~ma, ~mb))).astype(np.float32)


def contour(m: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask (both sides of every light/dark transition)."""
    e = np.zeros(m.shape, bool)
    dx = m[..., :, 1:] != m[..., :, :-1]
    e[..., :, 1:] |= dx
    e[..., :, :-1] |= dx
    dy = m[..., 1:, :] != m[..., :-1, :]
    e[..., 1:, :] |= dy
    e[..., :-1, :] |= dy
    return e


def dist_to(e: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) of every pixel to the nearest True pixel of e (H,W)."""
    if not e.any():
        return np.full(e.shape, float(max(e.shape)), np.float32)
    src = np.where(e, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)


def dist_batch(e: np.ndarray) -> np.ndarray:
    return np.stack([dist_to(e[i]) for i in range(e.shape[0])]) if len(e) else np.zeros(e.shape, np.float32)


def chamfer_sim(ea: np.ndarray, da: np.ndarray, eb: np.ndarray, db: np.ndarray, tau: float) -> np.ndarray:
    """Symmetric truncated chamfer similarity of silhouette contours, in [0, 1]."""
    na = int(ea.sum())
    nb = eb.sum(axis=(-2, -1))
    dbt = np.minimum(db, tau)
    dat = np.minimum(da, tau)
    d_ab = (dbt * ea[None]).sum(axis=(-2, -1)) / max(na, 1)
    d_ba = (dat[None] * eb).sum(axis=(-2, -1)) / np.maximum(nb, 1)
    d = 0.5 * (d_ab + d_ba)
    if na == 0:
        d = np.where(nb == 0, 0.0, tau)
    else:
        d = np.where(nb == 0, tau, d)
    return (1.0 - d / tau).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# gradient orientation histograms (HOG-like, unsigned, no block normalisation)
# ----------------------------------------------------------------------------------------------
def hog(x: np.ndarray, cell: int = 8, bins: int = 9):
    """x: (B,H,W) -> (normalised histograms (B,cy,cx,bins), cell gradient energy (B,cy,cx))."""
    x = blur(x, sigma=1.0, radius=2)
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[..., :, 1:-1] = x[..., :, 2:] - x[..., :, :-2]
    gy[..., 1:-1, :] = x[..., 2:, :] - x[..., :-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    b = np.minimum((ang * (bins / np.pi)).astype(np.int32), bins - 1)
    B, H, W = x.shape
    cy, cx = H // cell, W // cell
    mag = mag[:, : cy * cell, : cx * cell]
    b = b[:, : cy * cell, : cx * cell]
    cell_idx = (np.arange(cy * cell) // cell)[:, None] * cx + (np.arange(cx * cell) // cell)[None, :]
    idx = ((np.arange(B, dtype=np.int64)[:, None, None] * (cy * cx) + cell_idx[None]) * bins + b)
    hist = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=B * cy * cx * bins)
    hist = hist.reshape(B, cy, cx, bins).astype(np.float32)
    energy = hist.sum(axis=-1)
    hn = hist / (np.linalg.norm(hist, axis=-1, keepdims=True) + EPS)
    return hn, energy


def hog_sim(ha: np.ndarray, ea: np.ndarray, hb: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Orientation agreement in the cells where the target has structure (weights = target energy)."""
    cos = (ha[None] * hb).sum(axis=-1)  # (B, cy, cx)
    w = ea / (ea.sum() + EPS)
    return (cos * w[None]).sum(axis=(-2, -1)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# global balance / placement descriptors
# ----------------------------------------------------------------------------------------------
def balance(x: np.ndarray) -> np.ndarray:
    """(B,H,W) -> (B,9): mean, light centroid (x,y), dark centroid (x,y), light spread (x,y),
    left-right and top-bottom brightness differences."""
    B, H, W = x.shape
    xs = ((np.arange(W) + 0.5) / W).astype(np.float32)
    ys = ((np.arange(H) + 0.5) / H).astype(np.float32)
    m = x.mean(axis=(-2, -1))
    out = np.zeros((B, 9), np.float32)
    out[:, 0] = m
    for j, img in enumerate((x, 1.0 - x)):
        colsum = img.sum(axis=-2)  # (B, W)
        rowsum = img.sum(axis=-1)  # (B, H)
        tot = colsum.sum(axis=-1)
        ok = tot > 1e-3 * H * W
        cx = np.where(ok, (colsum * xs).sum(-1) / np.maximum(tot, EPS), 0.5)
        cy = np.where(ok, (rowsum * ys).sum(-1) / np.maximum(tot, EPS), 0.5)
        out[:, 1 + 2 * j] = cx
        out[:, 2 + 2 * j] = cy
        if j == 0:
            sx = np.sqrt(np.maximum((colsum * (xs[None] - cx[:, None]) ** 2).sum(-1) / np.maximum(tot, EPS), 0))
            sy = np.sqrt(np.maximum((rowsum * (ys[None] - cy[:, None]) ** 2).sum(-1) / np.maximum(tot, EPS), 0))
            out[:, 5] = np.where(ok, sx, 0.29)
            out[:, 6] = np.where(ok, sy, 0.29)
    out[:, 7] = x[:, :, : W // 2].mean(axis=(-2, -1)) - x[:, :, W - W // 2:].mean(axis=(-2, -1))
    out[:, 8] = x[:, : H // 2, :].mean(axis=(-2, -1)) - x[:, H - H // 2:, :].mean(axis=(-2, -1))
    return out


_BAL_W = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 1.0, 1.0], np.float32)


def balance_sim(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """fa (9,), fb (B,9) -> similarity in [0,1].

    Centroid terms are weighted by how much light (or dark) area both images actually have, so the
    position of a tiny or absent region does not dominate.
    """
    d = np.abs(fb - fa[None])
    w = np.broadcast_to(_BAL_W, d.shape).copy()
    light = np.minimum(fa[0], fb[:, 0])
    dark = np.minimum(1.0 - fa[0], 1.0 - fb[:, 0])
    lw = np.clip(light / 0.15, 0, 1)
    dw = np.clip(dark / 0.15, 0, 1)
    w[:, 1] *= lw
    w[:, 2] *= lw
    w[:, 5] *= lw
    w[:, 6] *= lw
    w[:, 3] *= dw
    w[:, 4] *= dw
    dist = (w * d).sum(axis=1) / w.sum(axis=1)
    return np.clip(1.0 - 2.0 * dist, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------------------------------
def quant_hash(u8: np.ndarray) -> bytes:
    """Hash of a thumbnail quantised to 16 levels: identical-looking frames share a hash."""
    return hashlib.blake2b((u8 >> 4).tobytes(), digest_size=8).digest()


def soft_mask(x: np.ndarray, t, softness: float = 0.04) -> np.ndarray:
    t = np.asarray(t, np.float32)
    if t.ndim == 1:
        t = t.reshape((-1,) + (1,) * (x.ndim - 1))
    return (1.0 / (1.0 + np.exp(-(x - t) / softness))).astype(np.float32)
