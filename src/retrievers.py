import os
import chromadb
import re
import json
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.environ.get("CHROMA_PATH", "../day6/results/chroma_db")
from rank_bm25 import BM25Okapi

def tok(s):
    return re.findall(r"\w+", s.lower())

class BM25Retriever:
    name = "bm25"
    def __init__(self, chunks_path="data/converted/chunks.json"):
        self.chunks = json.load(open(chunks_path, encoding="utf-8"))
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])

    def search(self, query, *, filters=None, top_k=5):
        scores = self.bm25.get_scores(tok(query))
        ok = lambda c: not filters or all(c["metadata"].get(k) == v for k, v in filters.items())
        idx = [i for i in scores.argsort()[::-1] if ok(self.chunks[i])][:top_k]
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(idx)]

class HybridRetriever:
    """Gộp dense + BM25 bằng Reciprocal Rank Fusion: score = Σ 1/(k + rank)."""
    name = "hybrid"
    def __init__(self, dense, bm25, k=60, pool=20):
        self.dense, self.bm25, self.k, self.pool = dense, bm25, k, pool

    def search(self, query, *, filters=None, top_k=5):
        fused = {}
        for ret in (self.dense, self.bm25):
            for r in ret.search(query, filters=filters, top_k=self.pool):
                e = fused.setdefault(r["id"], {**r, "score": 0.0})
                e["score"] += 1 / (self.k + r["rank"])
        out = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
        for i, r in enumerate(out):
            r["rank"] = i + 1
        return out
class DenseRetriever:
    name = "dense"
    def __init__(self, model="intfloat/multilingual-e5-small", collection="day06_B"):
        self.model = SentenceTransformer(model, device="cpu")
        self.col = chromadb.PersistentClient(CHROMA_PATH).get_collection(collection)

    def search(self, query, *, filters=None, top_k=5):
        vec = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        r = self.col.query(query_embeddings=[vec.tolist()], n_results=top_k, where=filters or None)
        return [{"id": i, "text": d, "source_id": m["document_id"], "source": m["source"],
                 "page": m["page"], "metadata": m, "score": 1 - dist, "rank": k + 1}
                for k, (i, d, m, dist) in enumerate(zip(r["ids"][0], r["documents"][0],
                                                        r["metadatas"][0], r["distances"][0]))]