# author hgh
# version 1.0
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, List

import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "onnx" / "bge_embedder"
ONNX_PATH = os.path.join(MODEL_DIR, "bge_embedder.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer")

session = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

app = FastAPI(title="Embedder ORT")


class EmbeddingRequest(BaseModel):
    input: Optional[List[str]] = None


@app.post("/v1/embeddings")
async def embed(req: EmbeddingRequest):
    if not req.input:
        return {"embeddings": []}

    try:
        enc = await asyncio.to_thread(
            lambda doc: tokenizer(doc, return_tensors="np", max_length=512, padding="max_length", truncation=True),
            req.input
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        embedding = await asyncio.to_thread(session.run, None,
                                            {"input_ids": input_ids, "attention_mask": attention_mask})
        print(f"embedding: {embedding}")
        return {"data": embedding[0].tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
