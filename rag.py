# rag.py
import os
from openai import OpenAI
from pinecone_utils import create_index_if_not_exists, get_index

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure index exists
create_index_if_not_exists(dim=1536)

# Connect to index
index = get_index()

# Embedding generator
def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# RAG Upsert
def upsert_document(doc_id: str, text: str, metadata=None):
    embedding = embed_text(text)
    index.upsert(vectors=[
        (doc_id, embedding, metadata or {"text": text})
    ])
    return {"status": "success", "id": doc_id}

# RAG Search
def search(query: str, top_k: int = 5):
    embedding = embed_text(query)
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = []
    for m in result.get("matches", []):
        matches.append({
            "id": m["id"],
            "score": m["score"],
            "text": m["metadata"].get("text", "")
        })

    return matches
