from pathlib import Path
import numpy as np

from embedder import Embedder
from gitsource import chunk_documents
from minsearch import VectorSearch, Index


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


embedder = Embedder()


# ==========================================================
# Q1. Embeddings
# ==========================================================

query = "How does approximate nearest neighbor search work?"

query_embedding = embedder.encode(query)

print(len(query_embedding))
print(query_embedding[0])


# ==========================================================
# Q2. Cosine similarity
# ==========================================================

doc = Path("../week2/07-sqlitesearch-vector.md").read_text(encoding="utf-8")

doc_embedding = embedder.encode(doc)

similarity = np.dot(query_embedding, doc_embedding)

print(similarity)


# ==========================================================
# Load all lessons
# ==========================================================

repo_dir = Path("../../llm-zoomcamp")

documents = []

for file in sorted(repo_dir.glob("*/lessons/*.md")):
    documents.append({
        "filename": str(file.relative_to(repo_dir)),
        "content": file.read_text(encoding="utf-8")
    })

print(len(documents))
print(documents[0]["filename"])
print(documents[-1]["filename"])

chunks = chunk_documents(
    documents,
    size=2000,
    step=1000
)

contents = [chunk["content"] for chunk in chunks]

X = embedder.encode_batch(contents)

print(X.shape)


# ==========================================================
# Q3. Search by hand
# ==========================================================

scores = X.dot(query_embedding)

best = np.argmax(scores)

print(scores[best])
print(chunks[best]["filename"])


# ==========================================================
# Build Vector Index
# ==========================================================

vector_index = VectorSearch()
vector_index.fit(X, chunks)


# ==========================================================
# Q4. Vector Search
# ==========================================================

query = "What metric do we use to evaluate a search engine?"

query_vector = embedder.encode(query)

results = vector_index.search(query_vector)

print(results[0]["filename"])


# ==========================================================
# Build Text Index
# ==========================================================

text_index = Index(
    text_fields=["content"]
)

text_index.fit(chunks)


# ==========================================================
# Q5. Text Search vs Vector Search
# ==========================================================

query = "How do I store vectors in PostgreSQL?"

text_results = text_index.search(
    query,
    num_results=5
)

query_vector = embedder.encode(query)

vector_results = vector_index.search(
    query_vector,
    num_results=5
)

print("TEXT SEARCH")

for r in text_results:
    print(r["filename"])

print()

print("VECTOR SEARCH")

for r in vector_results:
    print(r["filename"])


# ==========================================================
# Q6. Hybrid Search (RRF)
# ==========================================================

query = "How do I give the model access to tools?"

query_vector = embedder.encode(query)

vector_results = vector_index.search(
    query_vector,
    num_results=5
)

text_results = text_index.search(
    query,
    num_results=5
)

results = rrf([
    vector_results,
    text_results
])

print(results[0]["filename"])

print("\nTEXT SEARCH (Q6)")
for r in text_results:
    print(r["filename"])

print("\nVECTOR SEARCH (Q6)")
for r in vector_results:
    print(r["filename"])

print("\nRRF")
for r in results:
    print(r["filename"])