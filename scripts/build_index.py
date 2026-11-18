"""Precompute dense embeddings for all RAG chunks (run at Docker build time).

Reads  data/processed/docs.jsonl
Writes data/index/chunks.jsonl + data/index/embeddings.npy (L2-normalised float32)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402


def main():
    docs = [json.loads(l) for l in (config.DATA_DIR / "docs.jsonl").read_text().split("\n") if l]
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / "chunks.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs))

    from fastembed import TextEmbedding

    t0 = time.time()
    model = TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=config.EMBED_CACHE)
    texts = [f"{d['title']}\n{d['text']}" for d in docs]
    emb = np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    print(f"Indexed {len(docs)} chunks, dim={emb.shape[1]} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
