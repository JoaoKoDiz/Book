"""Optional semantic similarity with a pretrained vision model (feature extraction ONLY).

The model is used to *measure* similarity between a reference frame and a gameplay frame; it never
generates or modifies images. Supported (via Hugging Face `transformers`):

  * DINOv2   (default "facebook/dinov2-small"): CLS token + mean patch token, and a spatial
              patch-layout similarity (which kind of content is where).
  * CLIP     (e.g. "openai/clip-vit-base-patch32"): image embedding cosine similarity.

Requires `torch` and `transformers` and downloads the weights from the Hugging Face hub the first
time (~90 MB for dinov2-small). Without them the pipeline runs without semantic scores and the CSV /
report state that the semantic metric was not computed.
"""
from __future__ import annotations

import numpy as np

from .util import LOG


def availability() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on the environment
        return False, f"torch/transformers not installed ({e.__class__.__name__}: {e})"
    return True, "ok"


class SemanticEmbedder:
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "auto", batch: int = 32,
                 random_init: bool = False):
        import torch
        from transformers import AutoConfig, AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch = int(batch)
        self.torch = torch
        self.name = model_name
        self.kind = "clip" if "clip" in model_name.lower() else "dinov2"
        if random_init:  # used only by the software self-test (no download); scores are meaningless
            if self.kind == "clip":
                from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
                self.model = CLIPVisionModelWithProjection(CLIPVisionConfig(image_size=224, patch_size=32,
                                                                            hidden_size=64, intermediate_size=128,
                                                                            num_hidden_layers=2, num_attention_heads=2,
                                                                            projection_dim=32))
            else:
                from transformers import Dinov2Config, Dinov2Model
                self.model = Dinov2Model(Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                                      intermediate_size=128, patch_size=14, image_size=224))
        elif self.kind == "clip":
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        else:
            AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(device)
        if self.kind == "clip":
            self.mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
            self.std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
        else:
            self.mean = np.array([0.485, 0.456, 0.406], np.float32)
            self.std = np.array([0.229, 0.224, 0.225], np.float32)
        LOG.info("Semantic model %s loaded on %s%s.", model_name, device, " (RANDOM WEIGHTS, test only)" if random_init else "")

    def _prep(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (B, H, W, 3) uint8 -> (B, 3, H', W') float, letterboxed to a square for CLIP."""
        import cv2

        out = []
        for im in rgb:
            if self.kind == "clip":
                h, w = im.shape[:2]
                s = 224.0 / max(h, w)
                r = cv2.resize(im, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
                can = np.zeros((224, 224, 3), np.uint8)
                y0, x0 = (224 - r.shape[0]) // 2, (224 - r.shape[1]) // 2
                can[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
                im = can
            else:
                h, w = im.shape[:2]
                W = 224
                H = max(14, int(round(W * h / w / 14.0)) * 14)
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            x = (im.astype(np.float32) / 255.0 - self.mean) / self.std
            out.append(x.transpose(2, 0, 1))
        return np.stack(out)

    def embed(self, rgb: np.ndarray) -> dict:
        """Returns {'global': (B, D), 'patches': (B, 12, D) or None}, all L2-normalised float16.

        For DINOv2 the patch tokens are average-pooled to a 3 x 4 grid ("which kind of content is
        where"), small enough to cache for thousands of candidates.
        """
        torch = self.torch
        F_ = torch.nn.functional
        gl, pa = [], []
        with torch.inference_mode():
            for i in range(0, len(rgb), self.batch):
                x = torch.from_numpy(self._prep(rgb[i:i + self.batch])).to(self.device)
                o = self.model(pixel_values=x)
                if self.kind == "clip":
                    g = o.image_embeds
                    p = None
                else:
                    h = o.last_hidden_state  # (B, 1 + gh*gw, D)
                    gh, gw = x.shape[-2] // 14, x.shape[-1] // 14
                    tok = h[:, 1:1 + gh * gw]
                    g = torch.cat([F_.normalize(h[:, 0], dim=-1), F_.normalize(tok.mean(1), dim=-1)], dim=1)
                    grid = tok.reshape(tok.shape[0], gh, gw, -1).permute(0, 3, 1, 2)
                    p = F_.adaptive_avg_pool2d(grid, (3, 4)).flatten(2).transpose(1, 2)  # (B, 12, D)
                    p = F_.normalize(p, dim=-1)
                gl.append(F_.normalize(g, dim=-1).float().cpu().numpy().astype(np.float16))
                pa.append(None if p is None else p.float().cpu().numpy().astype(np.float16))
        return {"global": np.concatenate(gl), "patches": None if pa[0] is None else np.concatenate(pa)}


def semantic_similarity(ref_emb: dict, i: int, cand_emb: dict) -> np.ndarray:
    """Similarity in [0,1] of reference frame i to every candidate (global cosine, plus the spatial
    patch-layout agreement for DINOv2)."""
    g = cand_emb["global"].astype(np.float32) @ ref_emb["global"][i].astype(np.float32)
    s = (g + 1.0) / 2.0
    if ref_emb.get("patches") is not None and cand_emb.get("patches") is not None \
            and ref_emb["patches"].shape[1] == cand_emb["patches"].shape[1]:
        pl = (cand_emb["patches"].astype(np.float32) * ref_emb["patches"][i][None].astype(np.float32)).sum(-1).mean(-1)
        s = 0.5 * s + 0.5 * (pl + 1.0) / 2.0
    return s.astype(np.float32)
