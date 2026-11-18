"""Hybrid retriever: BM25 (lexical) + dense embeddings (semantic), fused with RRF.

Dense vectors are precomputed at Docker build time by scripts/build_index.py so the
container starts fast; only the query is embedded at request time.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

_TOKEN = re.compile(r"[a-z0-9؀-ۿ]+")
_STOP = set(
    "a an the and or of in on at to for with by from is are was were be been this that these "
    "those it its as what which who how do does did can i you we they me my your our their "
    "about any some there here have has had will would should could".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class HybridRetriever:
    def __init__(self) -> None:
        chunks_path = config.INDEX_DIR / "chunks.jsonl"
        self.chunks: list[dict] = [json.loads(l) for l in chunks_path.read_text().split("\n") if l]
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in self.chunks])
        emb_path = config.INDEX_DIR / "embeddings.npy"
        self.emb = np.load(emb_path) if emb_path.exists() else None

    @staticmethod
    @lru_cache(maxsize=1)
    def _embedder():
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=config.EMBED_CACHE)

    def _dense_scores(self, query: str) -> np.ndarray | None:
        if self.emb is None:
            return None
        q = np.array(next(iter(self._embedder().query_embed([query]))), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        return self.emb @ q

    def search(self, query: str, k: int = 6, source: str | None = None) -> list[dict]:
        mask = np.array(
            [source in (None, "any") or c["source"] == source for c in self.chunks], dtype=bool
        )
        if not mask.any():
            return []
        rrf = np.zeros(len(self.chunks), dtype=np.float32)
        bm = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        rankings = [bm]
        dense = self._dense_scores(query)
        if dense is not None:
            rankings.append(dense)
        for scores in rankings:
            scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            rrf += 1.0 / (60 + ranks)
        rrf = np.where(mask, rrf, -np.inf)
        out, seen_urls = [], {}
        for i in np.argsort(-rrf):
            c = self.chunks[int(i)]
            # at most 2 chunks per page so results cover more sources
            if seen_urls.get(c["url"], 0) >= 2:
                continue
            seen_urls[c["url"]] = seen_urls.get(c["url"], 0) + 1
            out.append({**c, "score": round(float(rrf[i]), 4)})
            if len(out) >= k:
                break
        return out


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
