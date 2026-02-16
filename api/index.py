from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

# ✅ CORS configuration (IMPORTANT for browser requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins
    allow_credentials=False,    # Must be False when using "*"
    allow_methods=["*"],        # Allow POST and OPTIONS
    allow_headers=["*"],        # Allow all headers
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request model
class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str


# Cosine similarity function
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# POST endpoint
@app.post("/similarity")
async def similarity(data: SimilarityRequest):

    if not data.docs or not data.query:
        raise HTTPException(status_code=400, detail="docs and query required")

    try:
        # Combine docs and query for batch embedding
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

        # Sort by similarity descending
        scores.sort(reverse=True, key=lambda x: x[0])

        top_3 = [doc for _, doc in scores[:3]]

        return {"matches": top_3}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
