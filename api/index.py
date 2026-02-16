from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

# ✅ Proper CORS for browser preflight
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.post("/similarity")
async def similarity(data: SimilarityRequest):

    inputs = data.docs + [data.query]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=inputs
    )

    embeddings = [item.embedding for item in response.data]

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    scores = []

    for i, doc_embed in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embed)
        scores.append((score, data.docs[i]))

    scores.sort(reverse=True, key=lambda x: x[0])

    top_3 = [doc for _, doc in scores[:3]]

    return {"matches": top_3}
