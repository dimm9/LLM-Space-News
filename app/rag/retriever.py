import os
import faiss
import pickle
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "spacenews.index")
META_PATH = os.path.join(DATA_DIR, "spacenews.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class NasaRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)

        self.cross_encoder = None
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
            print("Cross-Encoder loaded")
        except Exception as e:
            print(f"Failed to load Cross-Encoder: {e}")

        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError(f"No files in {DATA_DIR}. Run ingest.py first")

        self.index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        self.bm25_corpus = [self.tokenize(d['chunk']) for d in self.chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def tokenize(self, text: str):
        return re.findall(r"[a-z0-9]+", text.lower())

    def embed_texts(self, texts):
        return self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # hybrid search (Dense + BM25 + RRF)
    def retrieve_dense(self, query: str, k: int = 50):
        qv = self.embed_texts([query])
        scores, idxs = self.index.search(qv, k)
        return [(float(scores[0][i]), self.chunks[idxs[0][i]]) for i in range(len(idxs[0]))]

    def retrieve_bm25(self, query: str, k: int = 50):
        toks = self.tokenize(query)
        scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in idxs]

    def rrf_fuse(self, dense_results, sparse_results, k=60):
        scores = {}
        for rank, (score, doc) in enumerate(dense_results):
            # source+chunk_id - unique key
            doc_id = f"{doc['source']}_{doc['chunk_id']}"
            if doc_id not in scores: scores[doc_id] = {"score": 0, "doc": doc}
            scores[doc_id]["score"] += 1.0 / (k + rank)

        for rank, (score, doc) in enumerate(sparse_results):
            doc_id = f"{doc['source']}_{doc['chunk_id']}"
            if doc_id not in scores: scores[doc_id] = {"score": 0, "doc": doc}
            scores[doc_id]["score"] += 1.0 / (k + rank)

        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [(item["score"], item["doc"]) for item in sorted_items]

    def rerank(self, query: str, candidates):
        if self.cross_encoder:
            pairs = [(query, c[1]['chunk']) for c in candidates]
            scores = self.cross_encoder.predict(pairs)
            scored = [(float(s), c[1]) for s, c in zip(scores, candidates)]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        return candidates  #fallback if no model

    def pack_context(self, hits, max_per_source=2, max_chars=2000):
        per = {}
        ordered = []
        for _, rec in hits:
            key = (rec["source"], rec.get("page", 1))
            per.setdefault(key, 0)
            if per[key] < max_per_source:
                ordered.append(rec)
                per[key] += 1

        cites = []
        parts = []
        for i, rec in enumerate(ordered, start=1):
            cites.append({
                "n": i,
                "source": rec["source"],
                "page": rec.get("page", 1),
                "chunk_id": rec["chunk_id"]
            })
            parts.append(f"[{i}] {rec['chunk']}")

        ctx = "\n\n".join(parts)
        return ctx[:max_chars], cites

    def search(self, query: str, top_k: int = 5):
        dense = self.retrieve_dense(query, k=50)
        sparse = self.retrieve_bm25(query, k=50)

        fused = self.rrf_fuse(dense, sparse)

        top_candidates = fused[:20]
        reranked = self.rerank(query, top_candidates)

        return reranked[:top_k]


if __name__ == "__main__":
    rag = NasaRetriever()
    print("\n--- Test RAG ---")
    results = rag.search("Electron rocket")
    ctx, cites = rag.pack_context(results)
    print(ctx)
    print(cites)