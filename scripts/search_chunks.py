"""
Phase 2, Step 2: Given a user question, embed it and find the closest
pre-computed chunk vectors.

This is the part that runs on EVERY user query -- but notice it only
does one embedding call (for the question), not 39. The heavy lifting
(embedding the whole constitution) already happened once in
embed_chunks.py.
"""

import json
import os
from dotenv import load_dotenv
import numpy as np
import voyageai

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found -- check your .env file exists and has it set")
MODEL = "voyage-3"

client = voyageai.Client(api_key=VOYAGE_API_KEY)

with open("../data/processed/chunks_with_embeddings.json") as f:
    chunks = json.load(f)

chunk_vectors = np.array([c["embedding"] for c in chunks])  # shape: (39, dim)

def cosine_similarity(a, b):
    # a: single query vector, b: matrix of chunk vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm  # dot product of normalized vectors = cosine similarity

def search(query, top_k=3):
    # input_type="query" here (vs "document" when we embedded the chunks) --
    # Voyage's model treats questions and passages slightly differently
    # internally, which measurably improves retrieval quality. Small
    # detail, real effect.
    result = client.embed([query], model=MODEL, input_type="query")
    query_vec = np.array(result.embeddings[0])

    scores = cosine_similarity(query_vec, chunk_vectors)
    ranked_idx = np.argsort(-scores)[:top_k]

    print(f'\nQuery: "{query}"')
    results = []
    for i in ranked_idx:
        c = chunks[i]
        label = c.get("citation") or (f"Article {c['article']}" + (f", Section {c['section']}" if c["section"] else ""))
        print(f"  [{scores[i]:.3f}] {label}: {c['text'][:90]}...")
        results.append(c)
    return results

if __name__ == "__main__":
    import time
    search("what's the minimum number of senators needed to hold a meeting")
    time.sleep(21)  # stay under 3 requests/min while on Voyage's free (no-card) tier
    search("what happens if a senator misses too many meetings")
    time.sleep(21)
    search("can I send a campaign email through my Stevens account")
