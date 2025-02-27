from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

class RequestData(BaseModel):
    prevdata: List[str]  # Fix: Define as a list of strings
    text: str

@app.post("/process")
async def process_request(data: RequestData):
    past_requests = data.prevdata
    new_request = data.text

    if not past_requests:
        return {"error": "No past data provided"}

    # Convert past requests into vector embeddings
    past_embeddings = model.encode(past_requests)

    # Create FAISS index
    d = past_embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(past_embeddings)  # Fix: No need for np.array()

    # Convert new request to embedding
    new_embedding = model.encode([new_request])

    # Ensure new_embedding is in correct shape
    new_embedding = np.array(new_embedding).astype("float32")

    # Search for similar requests
    D, I = index.search(new_embedding, k=min(2, len(past_requests)))  # Avoid k > len(past_requests)

    # Convert L2 distance to similarity score
    similarity_scores = 1 / (1 + D)

    # Prepare response
    results = []
    for i, (score, idx) in enumerate(zip(similarity_scores[0], I[0])):
        results.append({
            "similar_request": past_requests[idx],
            "similarity": round(float(score), 2)
        })

    # Check similarity threshold
    threshold = 0.7
    flagged = all(score < threshold for score in similarity_scores[0])

    return {
        "new_request": new_request,
        "similar_requests": results,
        "flagged_for_review": flagged
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
