from typing import Optional,List
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import torch

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("/root/models/bge-small-loan", device=device)

class EmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = None
    input: Optional[List[str]] = None
    model: Optional[str] = None         
    encoding_format: Optional[str] = None 

@app.post("/v1/embeddings")
def embed(req: EmbeddingRequest):
    embeddings = model.encode(req.input, normalize_embeddings=True, batch_size=64)
    return {"data": [{"embedding": emb.tolist()} for emb in embeddings]}
