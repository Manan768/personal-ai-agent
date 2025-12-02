import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load .env file
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bse-embeddings")

def init_pinecone():
    if not PINECONE_API_KEY:
        raise EnvironmentError("Missing Pinecone API key")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

# Create index if not exists
def create_index_if_not_exists(dim=1536, metric="cosine"):
    pc = init_pinecone()
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"🆕 Created new index: {PINECONE_INDEX_NAME}")
    else:
        print(f"ℹ️ Index already exists: {PINECONE_INDEX_NAME}")

def get_index():
    pc = init_pinecone()
    index = pc.Index(PINECONE_INDEX_NAME)
    return index
