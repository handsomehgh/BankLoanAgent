# author hgh
# version 1.0
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI()

MODEL_PATH = "/root/models/bge-loan-reranker"
model = CrossEncoder(MODEL_PATH,max_length=512)

class RerankerRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]
    sorted_indices: List[int]

@app.post("/rerank",response_model=RerankResponse)
def rerank(req: RerankerRequest):
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs,show_progress_bar=False).tolist()
    sorted_scores = sorted(range(len(scores)),key=lambda k: scores[k],reverse=True)
    return RerankResponse(scores=scores,sorted_indices=sorted_scores)