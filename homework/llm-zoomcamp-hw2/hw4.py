from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from gitsource import (
    GithubRepositoryDataReader,
    chunk_documents,
)

from minsearch import (
    Index,
    VectorSearch,
)

from embedder import Embedder

import json
import pandas as pd

from evaluation_utils import llm_structured


# ==========================================================
# OpenAI
# ==========================================================

load_dotenv()

openai_client = OpenAI()


# ==========================================================
# Load course lessons
# ==========================================================

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]


# ==========================================================
# Ground Truth generation
# ==========================================================

class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


def generate_questions(doc):
    user_prompt = json.dumps(
        {
            "filename": doc["filename"],
            "content": doc["content"]
        },
        ensure_ascii=False
    )

    result, usage = llm_structured(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions
    )

    return result.questions, usage


# ==========================================================
# Q1. Generate questions for the first 3 lessons
# ==========================================================

usages = []

for doc in documents[:3]:
    print("=" * 80)
    print(doc["filename"])

    questions, usage = generate_questions(doc)

    usages.append(usage)

    for i, q in enumerate(questions, start=1):
        print(f"{i}. {q}")

    print(f"Input tokens: {usage.input_tokens}")
    print()

avg_input_tokens = sum(u.input_tokens for u in usages) / len(usages)

print(f"\nAverage input tokens: {avg_input_tokens:.2f}")


# ==========================================================
# Load Ground Truth (Q2-Q6)
# ==========================================================

ground_truth = pd.read_csv("ground-truth.csv").to_dict(orient="records")

print(f"\nGround Truth records: {len(ground_truth)}")
print(ground_truth[0])

# ==========================================================
# Chunk documents
# ==========================================================

embedder = Embedder()

chunks = chunk_documents(
    documents,
    size=2000,
    step=1000
)

print(f"\nChunks: {len(chunks)}")

contents = [chunk["content"] for chunk in chunks]

X = embedder.encode_batch(contents)

print(X.shape)

# ==========================================================
# Build Vector Index
# ==========================================================

vector_index = VectorSearch()
vector_index.fit(X, chunks)


# ==========================================================
# Build Text Index
# ==========================================================

text_index = Index(
    text_fields=["content"]
)

text_index.fit(chunks)

def text_search(query, num_results=5):
    return text_index.search(
        query,
        num_results=num_results
    )

def vector_search(query, num_results=5):
    query_vector = embedder.encode(query)

    return vector_index.search(
        query_vector,
        num_results=num_results
    )

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

def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)

    return rrf(
        [text_results, vector_results],
        k=k
    )

q = ground_truth[0]["question"]

print(q)

text_results = text_search(q)
vector_results = vector_search(q)

print(text_results[0]["filename"])
print(vector_results[0]["filename"])

# ==========================================================
# Evaluation
# ==========================================================

def compute_relevance(q, search_function):
    filename = q["filename"]

    results = search_function(q["question"])

    relevance = []

    for doc in results:
        relevance.append(int(doc["filename"] == filename))

    return relevance


def compute_relevance_total(ground_truth, search_function):
    relevance_total = []

    for q in ground_truth:
        relevance = compute_relevance(q, search_function)
        relevance_total.append(relevance)

    return relevance_total


def hit_rate(relevance):
    hits = 0

    for line in relevance:
        if 1 in line:
            hits += 1

    return hits / len(relevance)


def mrr(relevance):
    total = 0.0

    for line in relevance:
        for rank, value in enumerate(line):
            if value == 1:
                total += 1 / (rank + 1)
                break

    return total / len(relevance)


def evaluate(search_function):
    relevance = compute_relevance_total(
        ground_truth,
        search_function
    )

    return {
        "hit_rate": hit_rate(relevance),
        "mrr": mrr(relevance),
    }

print("\nText search evaluation")
print(evaluate(text_search))

print("\nVector search evaluation")
print(evaluate(vector_search))

print("\nHybrid search tuning")

for k in [1, 50, 100, 200]:
    result = evaluate(
        lambda query, k=k: hybrid_search(query, k=k)
    )

    print(f"k={k}: {result}")