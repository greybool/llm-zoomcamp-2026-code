# ==========================================================
# LLM Zoomcamp 2026
# Week 1 - Agentic RAG
#
# Homework Answers
# ----------------------------------------------------------
# Q1: 72 lesson pages
# Q2: 01-agentic-rag/lessons/14-agentic-loop.md
# Q3: ~7136 input tokens (≈7000)
# Q4: 295 chunks
# Q5: ~3x fewer input tokens after chunking
# Q6: ~3-4 search tool calls (homework answer: 4)
# ==========================================================


# ==========================================================
# Imports
# ==========================================================

from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents
from minsearch import Index

from dotenv import load_dotenv
from openai import OpenAI

from rag_helper import RAGBase

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import (
    OpenAIResponsesRunner,
    DisplayingRunnerCallback,
)


# ==========================================================
# Setup
# ==========================================================

load_dotenv()


# ==========================================================
# Q1. How many lesson pages
# ==========================================================

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

print(len(documents))  # Answer: 72


# ==========================================================
# Q4. Chunking
# ==========================================================

chunks = chunk_documents(
    documents,
    size=2000,
    step=1000
)

print(len(chunks))  # Answer: 295


# ==========================================================
# Q2. Indexing and searching
# ==========================================================

index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

index.fit(chunks)

results = index.search(
    query="How does the agentic loop keep calling the model until it stops?",
    num_results=5
)

print(results[0]["filename"])
# Answer:
# 01-agentic-rag/lessons/14-agentic-loop.md


# ==========================================================
# Q3. RAG
# ==========================================================

client = OpenAI()

rag = RAGBase(
    index=index,
    llm_client=client
)

query = "How does the agentic loop keep calling the model until it stops?"

answer, usage = rag.rag(query)

print(answer)
print()
print(usage)


# ==========================================================
# Q6. Agentic RAG
# ==========================================================

def search(query: str):
    """
    Search the course lessons.
    """

    print(f"SEARCH: {query}")

    return index.search(
        query=query,
        num_results=5
    )


agent_tools = Tools()
agent_tools.add_tool(search)

# Debug: useful to inspect the automatically generated tool schema.
# Can be removed after verifying the homework.
print(agent_tools.get_tools())


instructions = """
You're a course teaching assistant.

Answer the student's question using the search tool.

Make multiple searches with different keywords before answering.
"""


chat_interface = IPythonChatInterface()

callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)

result = runner.loop(
    prompt="How does the agentic loop work, and how is it different from plain RAG?",
    callback=callback,
)