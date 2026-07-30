# author hgh
# version 1.0
import asyncio
import os
from typing import List
import numpy as np
import onnxruntime as ort
from pathlib import Path
from fastapi import HTTPException

from fastapi import FastAPI
from pydantic import BaseModel
from scipy.special import expit
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "onnx" / "reranker"
ONNX_PATH = os.path.join(MODEL_DIR, "reranker.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer")

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

class RerankerRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]
    sorted_indices: List[int]

app = FastAPI(title="Reranker")

MAX_BATCH_SIZE = 128

def compute_scores_sync(query: str, documents: List[str]) -> List[float]:
    all_scores = []
    for i in range(0, len(documents), MAX_BATCH_SIZE):
        batch = documents[i:i+MAX_BATCH_SIZE]
        pairs = [[query, doc] for doc in batch]
        enc = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="np"
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        logits = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })[0]
        scores = expit(logits).flatten().tolist()
        all_scores.extend(scores)
    return all_scores

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankerRequest):
    if not req.documents:
        return RerankResponse(scores=[], sorted_indices=[])
    try:
        scores = await asyncio.to_thread(compute_scores_sync, req.query, req.documents)
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        return RerankResponse(scores=scores, sorted_indices=sorted_indices)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Reranking service error")