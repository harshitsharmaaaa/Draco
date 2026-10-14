"""ColBERT reranking with BGE-M3's multi-vector head.

Score = mean over query tokens of the max cosine similarity against passage
tokens (BGE-M3 / FlagEmbedding 'colbert' mode). It lies roughly in [0, 1] but is
a similarity, NOT a probability.
"""
import time

import torch

from app.ingestion import embed
from app.retrieval.retrievers import GPU_LOCK
from app.retrieval.schema import Candidate
from app.retrieval.timing import COLD_START_MS


class ColbertReranker:
    def __init__(self, model_id, device, share_backbone, batch_size, max_length, max_candidates):
        self.model_id, self.device = model_id, device
        self.share_backbone, self.batch_size = share_backbone, batch_size
        self.max_length, self.max_candidates = max_length, max_candidates
        self._loaded = False

    def _load(self):
        t0 = time.perf_counter()
        from huggingface_hub import hf_hub_download
        if self.share_backbone:
            st = embed.get_embedding_model()
            self.backbone, self.tokenizer = st[0].auto_model, st.tokenizer
            self.device = str(next(self.backbone.parameters()).device)
        else:
            from transformers import AutoModel, AutoTokenizer
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.backbone = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
        state = torch.load(hf_hub_download(self.model_id, "colbert_linear.pt"), map_location="cpu", weights_only=True)
        dim = state["weight"].shape[0]
        self.linear = torch.nn.Linear(self.backbone.config.hidden_size, dim)
        self.linear.load_state_dict(state)
        self.linear.to(self.device, dtype=next(self.backbone.parameters()).dtype).eval()
        self._loaded = True
        COLD_START_MS["colbert_load"] = round((time.perf_counter() - t0) * 1e3, 1)

    @torch.inference_mode()
    def _encode(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        hidden = self.backbone(**tok).last_hidden_state[:, 1:]  # drop CLS
        vecs = torch.nn.functional.normalize(self.linear(hidden), dim=-1)
        return vecs, tok["attention_mask"][:, 1:].bool()

    @torch.inference_mode()
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Raw scores aligned with `texts`. `.cpu()` forces completed GPU work."""
        with GPU_LOCK:
            if not self._loaded:
                self._load()
            q, qm = self._encode([query])
            q = q[0][qm[0]]  # (Lq, D)
            out = [0.0] * len(texts)
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketing
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                p, pm = self._encode([texts[i] for i in idx])
                sim = torch.einsum("qd,bpd->bqp", q, p).masked_fill(~pm[:, None, :], -1.0)
                scores = sim.max(dim=-1).values.mean(dim=-1).float().cpu().tolist()
                for i, sc in zip(idx, scores):
                    out[i] = sc
            return out

    def rerank(self, query: str, candidates: list[Candidate]) -> tuple[list[Candidate], dict]:
        pool = candidates[: self.max_candidates]  # already RRF ordered and unique
        scores = self.score(query, [c.text for c in pool])
        for c, s in zip(pool, scores):
            c.colbert_score = s
        ranked = sorted(pool, key=lambda c: (-c.colbert_score, c.rrf_rank or 0, c.chunk_id))
        n_batches = -(-len(pool) // self.batch_size)
        return ranked, {"reranked": len(pool), "skipped": len(candidates) - len(pool),
                        "batch_size": self.batch_size, "batches": n_batches}
