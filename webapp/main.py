import os
import openai

import os
import uuid
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# ------------------- Config -------------------
LM_BASE_URL = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:1234/v1")
LM_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL = os.getenv("OPENAI_MODEL", "phi-3-mini-4k-instruct")

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "wine_rag")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CSV_PATH = os.getenv("WINE_CSV_PATH", "wine-ratings.csv")

FORCE_REINDEX = os.getenv("FORCE_REINDEX", "0") == "1"
BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "128"))
# ------------------------------------------------


app = FastAPI()


# Local LLM client (LM Studio)
llm = OpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)

# Local embedder
embedder = SentenceTransformer(EMBED_MODEL)

# Local vector DB
qdrant = QdrantClient(url=QDRANT_URL)


class Body(BaseModel):
    query: str


@app.on_event("startup")
def startup_index():
    try:
        print("Startup: ensuring Qdrant index...")
        ensure_qdrant_index()
        print("Startup: Qdrant index ready.")
    except Exception as e:
        print("Startup indexing failed:", repr(e))
        raise

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post("/ask")
def ask(body: Body):
    hits = search(body.query, k=5)
    context = format_context(hits)
    answer = assistant(body.query, context)
    return {"response": answer, "hits": [h.payload for h in hits]}

# ------------------- Indexing -------------------

def ensure_qdrant_index():
    dim = embedder.get_sentence_embedding_dimension()

    # Create collection if missing
    try:
        qdrant.get_collection(QDRANT_COLLECTION)
        exists = True
    except Exception:
        exists = False

    if not exists:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    # Skip if already indexed (unless forced)
    if not FORCE_REINDEX:
        try:
            info = qdrant.get_collection(QDRANT_COLLECTION)
            # If it already has points, don’t re-index on every restart
            if getattr(info, "points_count", 0) and info.points_count > 0:
                return
        except Exception:
            pass

    # If forced, recreate cleanly
    if FORCE_REINDEX:
        qdrant.delete_collection(QDRANT_COLLECTION)
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Build text for embedding
    def row_to_text(r) -> str:
        parts = []
        for col in ['name','grape','region','variety','notes']:
            if col in r and pd.notna(r[col]):
                parts.append(f"{col}: {str(r[col]).strip()}")
        # Also include numeric fields if present
        for col in ["rating"]:
            if col in r and pd.notna(r[col]):
                parts.append(f"{col}: {r[col]}")
        return " | ".join(parts) if parts else str(r.to_dict())

    # Upsert in batches
    texts = []
    payloads = []
    for _, row in df.iterrows():
        r = row.to_dict()
        text = row_to_text(r)
        texts.append(text)
        payloads.append({
            "text": text,
            **{k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.items()}
        })

        if len(texts) >= BATCH_SIZE:
            upsert_batch(texts, payloads)
            texts, payloads = [], []

    if texts:
        upsert_batch(texts, payloads)


def upsert_batch(texts, payloads):
    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()
    points = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload=payloads[i],
        )
        for i in range(len(texts))
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search(query: str, k: int = 5):
    qvec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=qvec,
        limit=k,
        with_payload=True
    )
    hits = result.points
    return hits

def format_context(hits) -> str:
    # Keep context compact but informative
    chunks = []
    for i, h in enumerate(hits, start=1):
        p = h.payload or {}
        chunks.append(f"[{i}] {p.get('text', '')}")
    return "\n".join(chunks)


# ------------------- LLM Answering -------------------

def assistant(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a chatbot that helps users choose a wine based on their preferences."},
        {"role": "system", "content": f"Use this retrieved dataset context to answer:\n{context}"},
        {"role": "user", "content": query},
    ]

    resp = llm.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content