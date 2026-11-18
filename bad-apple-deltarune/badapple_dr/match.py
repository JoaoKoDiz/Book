"""Stages 4-7: retrieval, detailed comparison, temporal optimisation and targeted refinement.

Per reference frame ("target") a candidate pool is built:

  stage 4  exact coarse search over every database row             -> top `coarse_topk`
           temporal diversification                                -> `candidates` per target
  stage 5a detailed comparison at the mid level (database thumbnails)
  stage 5b the best `fine_topn` are RE-DECODED from the video at the fine level, rescored
           (multi-scale structure), optional semantic score
  stage 6  Viterbi over the pools (temporal.py)
  stage 7  refinement rounds:
             - second-chance wider search for the worst-matching targets
             - dense decoding of EVERY frame around the most promising candidates (frames the
               adaptive sampler skipped are recovered here), scored against nearby targets
             - re-optimisation; stop when the gain is negligible
Every round is checkpointed in work/match/.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import features as F
from .ffmpeg import fetch_frames, read_window
from .framestore import FrameStore
from .retrieval import coarse_scores, db_descriptors, diversify, retrieve, target_descriptors
from .scoring import STRUCT_KEYS, CandBatch, TargetSet, combine_total, score
from .temporal import optimise
from .util import LOG, Progress, atomic_save_npz, atomic_write_json, fingerprint, read_json

SCORE_KEYS = ("total", "composition", "structural", "semantic", "tone", "balance") + STRUCT_KEYS
ORIGIN_DB, ORIGIN_SECOND, ORIGIN_DENSE = 0, 1, 2
ORIGIN_NAMES = {ORIGIN_DB: "database", ORIGIN_SECOND: "database (second-chance search)",
                ORIGIN_DENSE: "dense refinement (frame not in database)"}


def empty_pool() -> dict:
    p = {"n": np.zeros(0, np.int64), "pts": np.zeros(0, np.float64), "row": np.zeros(0, np.int64),
         "origin": np.zeros(0, np.int8)}
    for k in SCORE_KEYS:
        p[k] = np.zeros(0, np.float32)
    return p


def pool_add(p: dict, n, pts, row, origin, sc: dict, sel=None) -> int:
    n = np.asarray(n, np.int64)
    idx = np.arange(len(n)) if sel is None else np.asarray(sel)
    have = set(p["n"].tolist())
    idx = [i for i in idx if int(n[i]) not in have]
    if not idx:
        return 0
    idx = np.array(idx)
    p["n"] = np.concatenate([p["n"], n[idx]])
    p["pts"] = np.concatenate([p["pts"], np.asarray(pts, np.float64)[idx]])
    p["row"] = np.concatenate([p["row"], np.asarray(row, np.int64)[idx]])
    p["origin"] = np.concatenate([p["origin"], np.full(len(idx), origin, np.int8)])
    for k in SCORE_KEYS:
        p[k] = np.concatenate([p[k], np.asarray(sc[k], np.float32)[idx]])
    return len(idx)


def pool_prune(p: dict, keep: int, must_keep_n=()) -> dict:
    order = np.argsort(-p["total"], kind="stable")
    sel = list(order[:keep])
    for n in must_keep_n:
        j = np.where(p["n"] == n)[0]
        if len(j) and j[0] not in sel:
            sel.append(int(j[0]))
    sel = np.array(sorted(sel), np.int64)
    return {k: v[sel] for k, v in p.items()}


def pack(pools: list[dict]) -> dict:
    lens = [len(p["n"]) for p in pools]
    d = {"offsets": np.cumsum([0] + lens).astype(np.int64)}
    for k in pools[0]:
        d[k] = np.concatenate([p[k] for p in pools])
    return d


def unpack(d) -> list[dict]:
    off = d["offsets"]
    keys = [k for k in d.keys() if k != "offsets"]
    return [{k: np.array(d[k][off[i]:off[i + 1]]) for k in keys} for i in range(len(off) - 1)]


def mid_from_fine(fine: np.ndarray) -> np.ndarray:
    return np.clip(np.round(F.pool2(fine.astype(np.float32))), 0, 255).astype(np.uint8)


class Matcher:
    def __init__(self, ffmpeg: str, cfg: dict, src_info: dict, db, ref: dict, work: Path):
        self.ffmpeg = ffmpeg
        self.cfg = cfg
        self.src = src_info
        self.db = db
        self.ref = ref
        self.dir = Path(work) / "match"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.w = cfg["weights"]
        self.ts = TargetSet(ref["fine"], self.w)
        self.T = self.ts.T
        self.td_all = target_descriptors(ref["mid"])  # coarse descriptors (dense-refinement prefilter)
        self.meta = db.meta
        self.db_pts = self.meta["pts"]
        self.db_n = self.meta["n"]
        self.fine_shape = tuple(ref["fine"].shape[1:])
        a = src_info["active_area"]
        self.crop = (a["x"], a["y"], a["w"], a["h"])
        v = src_info["video"]
        self.fps = float(v["fps_float"])
        self.file_start = float(src_info.get("start_time") or 0.0)
        self.store = FrameStore(Path(work) / "fine_cache", self.fine_shape)
        self.workers = int(cfg["rerank"]["workers"])
        self.stats = read_json(self.dir / "stats.json", {}) or {}
        for k in ("fine_fetched", "dense_frames_decoded", "dense_windows", "second_chance_targets",
                  "dense_frames_scored", "fetch_missing"):
            self.stats.setdefault(k, 0)
        self.dense_done: list[tuple[float, float]] = [tuple(x) for x in self.stats.get("dense_intervals", [])]
        self.sem = None
        self.sem_status = "disabled"
        self.ref_emb = None
        self.emb_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self._init_semantic()

    # ------------------------------------------------------------------------------------------
    def _reader_kw(self, pix_fmt="gray", size=None):
        Hf, Wf = self.fine_shape
        return {"pix_fmt": pix_fmt, "size": size or (Wf, Hf), "crop": self.crop}

    def _fp(self, extra: dict) -> str:
        base = {"rows": len(self.meta), "ref_n": self.T, "ref_sum": int(self.ref["fine"].astype(np.int64).sum()),
                "weights": self.w}
        base.update(extra)
        return fingerprint(base)

    def _save_stats(self):
        self.stats["dense_intervals"] = [list(x) for x in self.dense_done]
        atomic_write_json(self.dir / "stats.json", self.stats)

    # ------------------------------------------------------------------------------------------
    # semantic
    # ------------------------------------------------------------------------------------------
    def _init_semantic(self):
        sc = self.cfg["semantic"]
        mode = str(sc.get("mode", "auto")).lower()
        if mode in ("none", "off", "false", "0"):
            self.sem_status = "disabled by configuration"
            return
        from .semantic import SemanticEmbedder, availability

        ok, why = availability()
        if not ok:
            self.sem_status = f"not computed: {why}"
            if mode == "on":
                raise RuntimeError(f"semantic.mode=on but {why}. Install requirements-semantic.txt.")
            LOG.info("Semantic features skipped (%s).", why)
            return
        try:
            self.sem = SemanticEmbedder(sc["model"], sc.get("device", "auto"), int(sc.get("batch", 32)),
                                        random_init=bool(sc.get("_random_init_for_tests", False)))
            self.sem_status = f"computed with {sc['model']} on {self.sem.device}"
            if sc.get("_random_init_for_tests"):
                self.sem_status += " (RANDOM WEIGHTS - software test only)"
        except Exception as e:  # model download failure etc.
            self.sem = None
            self.sem_status = f"not computed: could not load {sc['model']} ({e.__class__.__name__}: {e})"
            if mode == "on":
                raise
            LOG.warning("Semantic model unavailable, continuing without it: %s", e)
            return
        cache = self.dir / "semantic_cache.npz"
        if cache.exists():
            d = np.load(cache)
            if str(d["model"]) == sc["model"]:
                pats = d["patches"] if d["has_patches"] else None
                for i, n in enumerate(d["n"]):
                    self.emb_cache[int(n)] = (d["global"][i], None if pats is None else pats[i])
        wpx = int(sc.get("image_width", 224))
        imgs = []
        for g in self.ref["fine"]:
            h = int(round(wpx * g.shape[0] / g.shape[1]))
            r = cv2.resize(g, (wpx, h), interpolation=cv2.INTER_LINEAR)
            imgs.append(np.repeat(r[..., None], 3, axis=2))
        self.ref_emb = self.sem.embed(np.stack(imgs))

    def _save_sem_cache(self):
        if self.sem is None or not self.emb_cache:
            return
        ns = np.array(sorted(self.emb_cache), np.int64)
        g = np.stack([self.emb_cache[int(n)][0] for n in ns])
        has_p = self.emb_cache[int(ns[0])][1] is not None
        p = np.stack([self.emb_cache[int(n)][1] for n in ns]) if has_p else np.zeros(1, np.float16)
        atomic_save_npz(self.dir / "semantic_cache.npz", n=ns, **{"global": g}, patches=p, has_patches=has_p,
                        model=np.array(self.cfg["semantic"]["model"]))

    def apply_semantic(self, pools: list[dict]):
        if self.sem is None:
            return
        n2pts = {}
        for p in pools:
            for n, t in zip(p["n"], p["pts"]):
                n2pts[int(n)] = float(t)
        need = sorted(set(n2pts) - set(self.emb_cache))
        if need:
            wpx = int(self.cfg["semantic"].get("image_width", 224))
            a = self.src["active_area"]
            hpx = max(14, int(round(wpx * a["h"] / a["w"] / 2.0)) * 2)
            prog = Progress(total=len(need), desc="Semantic features", unit="frames")
            for i in range(0, len(need), 512):
                part = need[i:i + 512]
                res = fetch_frames(self.ffmpeg, self.src["path"], [n2pts[n] for n in part], file_start=self.file_start,
                                   fps=self.fps, workers=self.workers, tolerance=self._tol(),
                                   **self._reader_kw("rgb24", (wpx, hpx)))
                got = [n for n in part if n2pts[n] in res]
                if got:
                    emb = self.sem.embed(np.stack([res[n2pts[n]][1] for n in got]))
                    for j, n in enumerate(got):
                        self.emb_cache[n] = (emb["global"][j], None if emb["patches"] is None else emb["patches"][j])
                prog.update(len(part))
            prog.close()
            self._save_sem_cache()
        for t, p in enumerate(pools):
            ok = np.array([int(n) in self.emb_cache for n in p["n"]], bool)
            if not ok.any():
                continue
            ce = {"global": np.stack([self.emb_cache[int(n)][0] for n in p["n"][ok]])}
            pt = [self.emb_cache[int(n)][1] for n in p["n"][ok]]
            ce["patches"] = None if pt[0] is None else np.stack(pt)
            sem = np.full(len(p["n"]), np.nan, np.float32)
            from .semantic import semantic_similarity

            sem[ok] = semantic_similarity(self.ref_emb, t, ce)
            p["semantic"] = sem
            p["total"] = combine_total(p["composition"], p["structural"], sem, self.w)

    # ------------------------------------------------------------------------------------------
    # fetching / scoring helpers
    # ------------------------------------------------------------------------------------------
    def _tol(self) -> float:
        return min(0.25 / self.fps, 0.004)

    def ensure_fine(self, ns, pts) -> None:
        missing = [(int(n), float(p)) for n, p in zip(ns, pts) if int(n) not in self.store]
        if not missing:
            return
        prog = Progress(total=len(missing), desc="Re-decoding candidate frames (fine level)", unit="frames")
        for i in range(0, len(missing), 1500):
            chunk = missing[i:i + 1500]
            res = fetch_frames(self.ffmpeg, self.src["path"], [p for _, p in chunk], file_start=self.file_start,
                               fps=self.fps, workers=self.workers, tolerance=self._tol(), progress=prog,
                               **self._reader_kw())
            items = []
            for n, p in chunk:
                if p in res:
                    items.append((n, res[p][0], res[p][1]))
                else:
                    self.stats["fetch_missing"] += 1
            self.store.add_many(items)
            self.stats["fine_fetched"] += len(items)
            self._save_stats()
        prog.close()
        if self.stats["fetch_missing"]:
            LOG.warning("%d candidate frames could not be re-decoded at their exact timestamp and were dropped.",
                        self.stats["fetch_missing"])

    def score_fine(self, t: int, ns) -> dict:
        fine = self.store.get_many(ns)
        return score(self.ts, t, CandBatch(mid_from_fine(fine), fine))

    def _mid_rank(self, t: int, rows: np.ndarray, top: int) -> np.ndarray:
        if len(rows) == 0:
            return rows
        srt = np.sort(rows)
        mids = np.asarray(self.db.thumbs[srt])
        sc = score(self.ts, t, CandBatch(mids))
        return srt[np.argsort(-sc["total"], kind="stable")[:top]]

    # ------------------------------------------------------------------------------------------
    # stage 4
    # ------------------------------------------------------------------------------------------
    def stage_retrieve(self):
        path = self.dir / "retrieval.npz"
        fp = self._fp({"retrieval": self.cfg["retrieval"]})
        if path.exists():
            d = np.load(path)
            if str(d["fp"]) == fp:
                LOG.info("Stage 4: using saved retrieval results.")
                return d["rows"], d["scores"]
        LOG.info("Stage 4: exact coarse retrieval of %d targets over %s database frames.", self.T, f"{len(self.meta):,}")
        t0 = time.time()
        rows, scores = retrieve(self.db, self.ref["mid"], self.ts.uniform, self.cfg)
        atomic_save_npz(path, rows=rows, scores=scores, fp=np.array(fp))
        self.stats["retrieval_seconds"] = time.time() - t0
        self._save_stats()
        return rows, scores

    # ------------------------------------------------------------------------------------------
    # stage 5
    # ------------------------------------------------------------------------------------------
    def stage_rerank(self, rows_all, scores_all) -> list[dict]:
        path = self.dir / "pools_round0.npz"
        fp = self._fp({"retrieval": self.cfg["retrieval"], "rerank": self.cfg["rerank"], "sem": self.sem_status})
        if path.exists():
            d = np.load(path)
            if str(d["fp"]) == fp:
                LOG.info("Stage 5: using saved detailed-comparison results.")
                return unpack({k: d[k] for k in d.files if k != "fp"})
        rc = self.cfg["retrieval"]
        topn = int(self.cfg["rerank"]["fine_topn"])
        LOG.info("Stage 5a: detailed comparison (mid level) of up to %d diversified candidates per target.",
                 int(rc["candidates"]))
        prog = Progress(total=self.T, desc="Detailed comparison", unit="targets")
        ranked = []
        for t in range(self.T):
            r, _ = diversify(rows_all[t], scores_all[t], self.db_pts, int(rc["candidates"]), float(rc["nms_window"]),
                             int(rc["max_per_window"]))
            ranked.append(self._mid_rank(t, r, topn))
            prog.update(1)
        prog.close()
        allrows = np.unique(np.concatenate(ranked)) if ranked else np.zeros(0, np.int64)
        LOG.info("Stage 5b: %s unique finalists -> re-decoding them from the video at %dx%d.",
                 f"{len(allrows):,}", self.fine_shape[1], self.fine_shape[0])
        self.ensure_fine(self.db_n[allrows], self.db_pts[allrows])
        pools = []
        for t in range(self.T):
            r = np.array([x for x in ranked[t] if int(self.db_n[x]) in self.store], np.int64)
            p = empty_pool()
            if len(r):
                sc = self.score_fine(t, self.db_n[r])
                pool_add(p, self.db_n[r], self.db_pts[r], r, ORIGIN_DB, sc)
            pools.append(p)
        self.apply_semantic(pools)
        atomic_save_npz(path, fp=np.array(fp), **pack(pools))
        return pools

    # ------------------------------------------------------------------------------------------
    # stage 6
    # ------------------------------------------------------------------------------------------
    def stage_optimise(self, pools: list[dict]):
        cands = []
        for p in pools:
            if len(p["n"]) == 0:
                raise RuntimeError("A reference frame has no candidates -- the database is empty?")
            fine = self.store.get_many(p["n"])
            g16 = F.pyramid(F.to_float(fine), 4)[3]
            cands.append({"total": p["total"], "n": p["n"], "pts": p["pts"], "g16": g16})
        path, coh, pc, dref = optimise(cands, self.ts.g16(), self.cfg["temporal"])
        return {"path": path, "coherence": coh, "pair_cost": pc, "dref": dref,
                "mean_total": float(np.mean([pools[t]["total"][path[t]] for t in range(self.T)])),
                "mean_best": float(np.mean([pools[t]["total"].max() for t in range(self.T)]))}

    # ------------------------------------------------------------------------------------------
    # stage 7
    # ------------------------------------------------------------------------------------------
    def _second_chance(self, pools, sel, rows_all, scores_all):
        rcfg = self.cfg["refine"]
        rc = self.cfg["retrieval"]
        totals = np.array([pools[t]["total"][sel["path"][t]] for t in range(self.T)])
        k = int(math.ceil(float(rcfg["worst_fraction"]) * self.T))
        worst = np.argsort(totals)[:k]
        LOG.info("Stage 7a: second-chance search for the %d worst-matching frames (wider candidate lists).", k)
        add = {}
        for t in worst:
            r, _ = diversify(rows_all[t], scores_all[t], self.db_pts, int(rcfg["extra_candidates"]),
                             float(rc["nms_window"]), int(rc["max_per_window"]) + 2)
            have = set(pools[t]["n"].tolist())
            r = np.array([x for x in r if int(self.db_n[x]) not in have], np.int64)
            add[int(t)] = self._mid_rank(int(t), r, int(rcfg["extra_fine"]))
        allrows = np.unique(np.concatenate(list(add.values()))) if add else np.zeros(0, np.int64)
        self.ensure_fine(self.db_n[allrows], self.db_pts[allrows])
        for t, r in add.items():
            r = np.array([x for x in r if int(self.db_n[x]) in self.store], np.int64)
            if len(r):
                pool_add(pools[t], self.db_n[r], self.db_pts[r], r, ORIGIN_SECOND, self.score_fine(t, self.db_n[r]))
        self.stats["second_chance_targets"] += len(add)

    def _covered(self, a: float, b: float) -> bool:
        return any(x <= a + 1e-6 and b <= y + 1e-6 for x, y in self.dense_done)

    def _dense(self, pools, sel):
        rcfg = self.cfg["refine"]
        W = float(rcfg["window"])
        totals = np.array([pools[t]["total"][sel["path"][t]] for t in range(self.T)])
        centers = []
        for t in np.argsort(totals):  # worst targets first
            p = pools[t]
            order = np.argsort(-p["total"])
            chosen = []
            for j in [sel["path"][t]] + list(order):
                pt = float(p["pts"][j])
                if all(abs(pt - c) > W for c in chosen):
                    chosen.append(pt)
                if len(chosen) >= int(rcfg["per_target"]):
                    break
            centers += [(c, int(t)) for c in chosen]
        # merge into windows, keep the priority order of the first center of each window
        wins: list[dict] = []
        for c, t in sorted(centers):
            a, b = c - W, c + W
            if wins and a <= wins[-1]["b"]:
                wins[-1]["b"] = max(wins[-1]["b"], b)
                wins[-1]["targets"].add(t)
            else:
                wins.append({"a": a, "b": b, "targets": {t}})
        prio = {t: i for i, t in enumerate(np.argsort(totals))}
        wins.sort(key=lambda w: min(prio[t] for t in w["targets"]))
        budget = int(rcfg["max_frames_per_round"])
        todo, used = [], 0
        for w in wins:
            if self._covered(w["a"], w["b"]):
                continue
            row0 = self.db.rows_before(w["a"])
            w["row0"] = row0
            w["start"] = float(self.db_pts[row0])
            est = int((w["b"] - w["start"]) * self.fps) + 1
            if used + est > budget:
                continue
            used += est
            todo.append(w)
        if not todo:
            LOG.info("Stage 7b: no new intervals to decode densely.")
            return 0
        LOG.info("Stage 7b: decoding %d intervals densely (~%s frames, every frame incl. those skipped by sampling).",
                 len(todo), f"{used:,}")
        neigh = int(rcfg["neighbours"])
        prefilter = int(rcfg.get("prefilter", 24))
        prog = Progress(total=len(todo), desc="Dense refinement", unit="intervals")
        added = 0

        def decode(w):
            pts, fr = read_window(self.ffmpeg, self.src["path"], w["start"], w["b"], file_start=self.file_start,
                                  fps=self.fps, threads=2, **self._reader_kw())
            return w, pts, fr

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for w, pts, fr in ex.map(decode, todo):
                prog.update(1)
                self.dense_done.append((w["start"], w["b"]))
                if fr is None or len(pts) == 0:
                    continue
                n0 = int(self.db_n[w["row0"]])
                if abs(pts[0] - w["start"]) < 1e-4:
                    ns = n0 + np.arange(len(pts), dtype=np.int64)
                else:  # should not happen; fall back to constant-frame-rate numbering
                    LOG.warning("Dense window at %.3fs did not start on the expected frame; using CFR numbering.", w["start"])
                    ns = n0 + np.round((pts - w["start"]) * self.fps).astype(np.int64)
                self.stats["dense_frames_decoded"] += len(pts)
                self.stats["dense_windows"] += 1
                mids = mid_from_fine(fr)
                cb = CandBatch(mids, fr)
                tset = set()
                for t in w["targets"]:
                    tset.update(range(max(0, t - neigh), min(self.T, t + neigh + 1)))
                tl = sorted(tset)
                # cheap exact prefilter (same coarse score as stage 4) -> full scoring of the best few
                S = coarse_scores({k: v[tl] for k, v in self.td_all.items()}, db_descriptors(mids), self.w,
                                  self.ts.uniform[tl])
                new_items = {}
                for ii, t in enumerate(tl):
                    cand = np.argsort(-S[ii], kind="stable")[:prefilter]
                    sc = score(self.ts, t, cb.subset(cand))
                    order = np.argsort(-sc["total"])
                    have = set(pools[t]["n"].tolist())
                    floor = float(np.sort(pools[t]["total"])[-min(len(have), 8)]) if have else -1.0
                    keep = [j for j in order[:6] if sc["total"][j] > floor and int(ns[cand[j]]) not in have]
                    if keep:
                        for j in keep:
                            new_items[int(ns[cand[j]])] = (float(pts[cand[j]]), fr[cand[j]])
                        added += pool_add(pools[t], ns[cand], pts[cand], np.full(len(cand), -1), ORIGIN_DENSE, sc,
                                          sel=keep)
                    self.stats["dense_frames_scored"] += len(cand)
                self.store.add_many([(n, p, f) for n, (p, f) in new_items.items()])
        prog.close()
        self._save_stats()
        return added

    def run(self) -> tuple[list[dict], dict]:
        rows_all, scores_all = self.stage_retrieve()
        pools = self.stage_rerank(rows_all, scores_all)
        sel = self.stage_optimise(pools)
        LOG.info("Stage 6: mean similarity %.4f (per-frame best %.4f).", sel["mean_total"], sel["mean_best"])
        rcfg = self.cfg["refine"]
        fp = self._fp({"retrieval": self.cfg["retrieval"], "rerank": self.cfg["rerank"], "refine": rcfg,
                       "temporal": self.cfg["temporal"], "sem": self.sem_status})
        prog_path = self.dir / "refine_progress.json"
        state = read_json(prog_path, {}) or {}
        start_round = 0
        history = [{"round": 0, "mean_total": sel["mean_total"], "mean_best": sel["mean_best"]}]
        if state.get("fp") == fp and state.get("rounds_done", 0) > 0:
            k = int(state["rounds_done"])
            d = np.load(self.dir / f"pools_round{k}.npz")
            pools = unpack({kk: d[kk] for kk in d.files if kk != "fp"})
            sel = self.stage_optimise(pools)
            start_round = k
            history = state.get("history", history)
            LOG.info("Stage 7: resuming after refinement round %d (mean similarity %.4f).", k, sel["mean_total"])
        for rnd in range(start_round, int(rcfg["rounds"])):
            t0 = time.time()
            before = sel["mean_total"]
            if rnd == 0:
                self._second_chance(pools, sel, rows_all, scores_all)
            added = self._dense(pools, sel)
            self.apply_semantic(pools)
            keep = int(self.cfg["rerank"]["fine_topn"]) + 8
            pools = [pool_prune(p, keep, [p["n"][sel["path"][t]]]) for t, p in enumerate(pools)]
            sel = self.stage_optimise(pools)
            gain = sel["mean_total"] - before
            history.append({"round": rnd + 1, "mean_total": sel["mean_total"], "mean_best": sel["mean_best"],
                            "dense_candidates_added": added, "seconds": time.time() - t0})
            LOG.info("Stage 7 round %d: mean similarity %.4f -> %.4f (gain %+.4f).", rnd + 1, before,
                     sel["mean_total"], gain)
            atomic_save_npz(self.dir / f"pools_round{rnd + 1}.npz", fp=np.array(fp), **pack(pools))
            atomic_write_json(prog_path, {"fp": fp, "rounds_done": rnd + 1, "history": history})
            self._save_stats()
            if gain < float(rcfg["min_gain"]) and rnd > 0:
                LOG.info("Stage 7: gain below %.4f, stopping refinement.", float(rcfg["min_gain"]))
                break
        self.stats["refine_history"] = history
        self.stats["semantic_status"] = self.sem_status
        self._save_stats()
        return pools, sel
