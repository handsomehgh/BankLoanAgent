# author hgh
# version 1.0
import logging
import os
from pathlib import Path
from typing import Optional, List

import onnxruntime as ort
from fastapi import FastAPI
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
def embed(req: EmbeddingRequest):
    if not req.input:
        return {"embeddings": []}

    enc = tokenizer(
        req.input,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    embedding = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    return {"data": embedding.tolist()}
