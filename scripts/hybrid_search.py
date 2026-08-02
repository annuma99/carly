"""
Phase 3: Hybrid retrieval.

Combines two independent rankings of the same 39 chunks:
  1. BM25 (keyword/exact-term match -- pure local math, no API call)
  2. Semantic search via Voyage embeddings (meaning-based -- one API call
     to embed the query, then compare against our cached chunk vectors)

Then merges the two ranked lists with Reciprocal Rank Fusion (RRF):
each chunk's score = 1/(k + rank_in_list_A) + 1/(k + rank_in_list_B)
A chunk that ranks well in EITHER list gets rewarded; a chunk that ranks
well in BOTH gets rewarded the most. This avoids having to compare BM25's
unbounded scores directly against cosine similarity's 0-1 scores --
we only ever look at RANK POSITION, not raw score, which is what makes
combining two totally different scoring systems possible.
"""

import json
import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import voyageai

load_dotenv()
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found -- check your .env file")

client = voyageai.Client(api_key=VOYAGE_API_KEY)
MODEL = "voyage-3"
RRF_K = 60  # standard RRF constant; dampens the impact of very top ranks

with open("data/processed/chunks_with_embeddings.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
chunk_vectors = np.array([c["embedding"] for c in chunks])

# ---- Set up BM25 (local, no API) ----
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

# ---- Helper: cosine similarity ----
def cosine_similarity(query_vec, matrix):
    q_norm = query_vec / np.linalg.norm(query_vec)
    m_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m_norm @ q_norm

def hybrid_search(query, top_k=3):
    # --- Ranking A: BM25 ---
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = np.argsort(-bm25_scores)  # chunk indices, best first

    # --- Ranking B: semantic (one API call) ---
    result = client.embed([query], model=MODEL, input_type="query")
    query_vec = np.array(result.embeddings[0])
    sem_scores = cosine_similarity(query_vec, chunk_vectors)
    sem_ranked = np.argsort(-sem_scores)

    # --- Reciprocal Rank Fusion ---
    rrf_scores = {}
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(sem_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)

    final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    print(f'\nQuery: "{query}"')
    for idx, score in final_ranked[:top_k]:
        c = chunks[idx]
        label = f"Article {c['article']}" + (f", Section {c['section']}" if c["section"] else "")
        print(f"  [RRF {score:.4f}] {label}: {c['text'][:90]}...")

if __name__ == "__main__":
    hybrid_search("what is a Recognized Student Organization")
    import time; time.sleep(21)
    hybrid_search("Section 4.2 disciplinary actions")
    time.sleep(21)
    hybrid_search("what's the minimum number of senators needed to hold a meeting")