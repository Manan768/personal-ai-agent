import math
import requests
from memory import memory
from rag import search as rag_search, upsert_document as rag_upsert
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import os
import datetime

# INITIALIZE PINECONE

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(host=os.getenv("PINECONE_HOST"))

# Embedding model
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")


# 1. RAG UPSERT (store text)

def rag_upsert(doc_id: str, text: str):
    try:
        embedding = embed_model.embed_query(text)

        index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": {"text": text}
            }]
        )

        return {"status": "success", "doc_id": doc_id}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# 2. RAG SEARCH (retrieve)

def rag_search(query: str):
    try:
        embedding = embed_model.embed_query(query)

        result = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )

        matches = []
        for m in result.get("matches", []):
            matches.append({
                "id": m["id"],
                "score": m.get("score"),
                "text": m["metadata"]["text"]
            })

        return matches

    except Exception as e:
        return [{"error": str(e)}]

def rag_query(query: str):
    results = rag_search(query)
    if not results:
        return "No relevant results found."
    
    best = results[0]
    return f"Top result (score={best['score']}): {best['text']}"

def rag_add(payload: str):
    """
    Expected format:
    "doc_id::content"
    """
    if "::" not in payload:
        return "Format error. Use: id::text"

    doc_id, text = payload.split("::", 1)
    rag_upsert(doc_id.strip(), text.strip())
    return "Document added to knowledge base."


# 1️⃣ Calculator Tool
def calculator(expression: str):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {e}"


# 2️⃣ Web Search Tool (DuckDuckGo Instant API)
def web_search(query: str):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1"
        r = requests.get(url).json()
        if r.get("AbstractText"):
            return r["AbstractText"]
        else:
            return "No summary found."
    except:
        return "Search failed."


# 3️⃣ System Info Tool
def system_info(_=None):
    return "This is Personal AI Agent (Python LangGraph v1)."


# 4️⃣ Memory Inspector Tool
def list_memory(_=None):
    data = memory.all()
    if not data:
        return "No stored memory."
    
    return "\n".join([f"{k}: {v}" for k, v in data.items()])

from memory import memory

def memory_lookup(key: str):
    """Lookup a memory key and return its value or a friendly message."""
    key = key.strip().lower()
    mem = memory.all()
    return mem.get(key, f"No memory found for key '{key}'")


# 🆕 New Tool: Calendar / Time
def get_calendar(_=None):
    """
    Returns the current date and time.
    Real implementation could connect to Google Calendar API here.
    """
    now = datetime.datetime.now()
    return f"Current Date & Time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

TOOLS = {
    "calculator": calculator,
    "search": web_search,
    "system_info": system_info,
    "list_memory": list_memory,
    "memory_lookup": memory_lookup,
    "rag_query": rag_query,
    "rag_add": rag_add,
    "calendar": get_calendar,
}
