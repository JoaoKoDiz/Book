"""Stage 6 -- temporal optimisation of the whole sequence (Viterbi / dynamic programming).

Minimises   sum_t (1 - S_t(c_t))  +  sum_t P(c_{t-1}, c_t)

P = lambda_rhythm * | change(output t-1 -> t) - change(reference t-1 -> t) |
  + lambda_jump   * jump(c_{t-1}, c_t) * (1 - reference_cut_t)

* the rhythm term makes the output change when (and as much as) the reference changes:
  it suppresses flicker on static reference frames and preserves the animation's timing;
* the jump term prefers continuing in the same gameplay scene (same frame / next frames: 0,
  same scene within `scene_seconds`: 0.25, unrelated part of the video: 1) unless the reference
  itself cuts at that moment.
Accuracy stays primary: a frame can only use candidates whose similarity is within `max_drop` of the
best candidate for that frame, so coherence can never cost more than that per frame.
"""
from __future__ import annotations

import numpy as np


def _pair_cost(a: dict, b: dict, dref: float, tc: dict) -> np.ndarray:
    dout = np.abs(a["g16"][:, None] - b["g16"][None]).mean(axis=(-2, -1))
    rhythm = np.abs(dout - dref)
    dn = b["n"][None, :] - a["n"][:, None]
    cont = (dn >= 0) & (dn <= 3)
    near = np.abs(b["pts"][None, :] - a["pts"][:, None]) <= float(tc["scene_seconds"])
    jump = np.where(cont, 0.0, np.where(near, 0.25, 1.0))
    refcut = min(1.0, dref / max(1e-6, float(tc["ref_cut"])))
    return (float(tc["lambda_rhythm"]) * rhythm + float(tc["lambda_jump"]) * jump * (1.0 - refcut)).astype(np.float64)


def optimise(cands: list[dict], ref_g16: np.ndarray, tc: dict):
    """cands[t]: dict with arrays total (K,), n (K,), pts (K,), g16 (K,12,16).

    Returns (choice index per t, temporal coherence per t, pair cost per t, dref per t).
    """
    T = len(cands)
    max_drop = float(tc["max_drop"])
    dref = np.zeros(T)
    if T > 1:
        dref[1:] = np.abs(ref_g16[1:] - ref_g16[:-1]).mean(axis=(-2, -1))
    unary = []
    for c in cands:
        tot = c["total"].astype(np.float64)
        ok = tot >= tot.max() - max_drop
        unary.append(np.where(ok, 1.0 - tot, np.inf))
    cost = unary[0].copy()
    back = [np.zeros(len(unary[0]), np.int64)]
    for t in range(1, T):
        P = _pair_cost(cands[t - 1], cands[t], dref[t], tc)
        tot = cost[:, None] + P
        arg = np.argmin(tot, axis=0)
        cost = tot[arg, np.arange(tot.shape[1])] + unary[t]
        back.append(arg)
    path = [int(np.argmin(cost))]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path = path[::-1]
    pc = np.zeros(T)
    for t in range(1, T):
        pc[t] = _pair_cost({k: v[path[t - 1]:path[t - 1] + 1] for k, v in cands[t - 1].items() if k in ("g16", "n", "pts")},
                           {k: v[path[t]:path[t] + 1] for k, v in cands[t].items() if k in ("g16", "n", "pts")},
                           dref[t], tc)[0, 0]
    ref_scale = float(tc["lambda_rhythm"]) * 0.25 + float(tc["lambda_jump"])
    coherence = 1.0 - np.clip(pc / max(ref_scale, 1e-9), 0.0, 1.0)
    coherence[0] = 1.0
    return path, coherence, pc, dref
