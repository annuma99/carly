"""
Phase 2, Step 1: Embed all chunks ONCE and cache the vectors.

Run this whenever chunks.json changes (new document added, chunk edited).
Do NOT run this on every user query -- that's the whole point of caching.

Requires: pip install voyageai
Get an API key at https://www.voyageai.com/ (they have a free tier that
is more than enough for a corpus this small).
"""

import json
import os
from dotenv import load_dotenv
import voyageai

load_dotenv()  # reads the .env file and loads its keys into os.environ

# ---- Config ----
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found -- check your .env file exists and has it set")
MODEL = "voyage-3"          # good general-purpose embedding model
INPUT_TYPE = "document"     # Voyage distinguishes "document" vs "query" embeddings
                            # -- documents and queries are embedded slightly
                            # differently under the hood for better retrieval

client = voyageai.Client(api_key=VOYAGE_API_KEY)

with open("../data/processed/chunks.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# ---- Rate-limit-safe batching ----
# Voyage caps free (no-payment-method) accounts at 3 requests/min and
# 10K tokens/min. Our 39 chunks (~9-10K tokens combined) sit right at
# that ceiling, so we split into small batches and pause between calls
# to stay under it. If you've added a payment method on Voyage's
# dashboard, you can skip this and just call client.embed(texts, ...)
# directly in one shot -- see the comment below.
import time

BATCH_SIZE = 10   # keeps each batch comfortably under 10K tokens/min
DELAY_SECONDS = 21  # (60s / 3 RPM) rounded up, so we never exceed 3 req/min

embeddings = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    result = client.embed(batch, model=MODEL, input_type=INPUT_TYPE)
    embeddings.extend(result.embeddings)
    print(f"Embedded chunks {i} to {i + len(batch) - 1}")
    if i + BATCH_SIZE < len(texts):
        time.sleep(DELAY_SECONDS)

# If you've added a payment method and have standard rate limits, replace
# the loop above with:
#   result = client.embed(texts, model=MODEL, input_type=INPUT_TYPE)
#   embeddings = result.embeddings

print(f"Embedded {len(embeddings)} chunks")
print(f"Vector dimensionality: {len(embeddings[0])}")

# Attach each vector to its chunk and save. Now the embedding travels
# with its metadata (document/article/section) -- retrieval in Phase 3
# will need both the vector (to search) and the metadata (to cite).
for chunk, vec in zip(chunks, embeddings):
    chunk["embedding"] = vec

with open("../data/processed/chunks_with_embeddings.json", "w") as f:
    json.dump(chunks, f)

print("Saved to chunks_with_embeddings.json")