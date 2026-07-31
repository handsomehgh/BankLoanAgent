# author hgh
# version 1.0
# merged_server.py
import asyncio
import os
import logging
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.special import expit
from transformers import AutoTokenizer

# ==================== 日志配置 ====================
LOG_DIR = "/root/autodl-tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def _setup_logger(name: str, filename: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(LOG_DIR, filename))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.propagate = False
    return logger

logger_embed = _setup_logger("embedder", "embedder.log")
logger_rerank = _setup_logger("reranker", "reranker.log")
logger_context = _setup_logger("context", "context.log")

# ==================== 全局模型变量 ====================
emb_session = None
emb_tokenizer = None
rerank_session = None
rerank_tokenizer = None
context_session = None
context_tokenizer = None

def load_models():
    global emb_session, emb_tokenizer, rerank_session, rerank_tokenizer, context_session, context_tokenizer
    # Embedder
    emb_session = ort.InferenceSession(
        "/root/autodl-tmp/models/embedder/bge_embedder.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    emb_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/embedder/tokenizer")
    # Reranker
    rerank_session = ort.InferenceSession(
        "/root/autodl-tmp/models/reranker/reranker.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    rerank_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/reranker/tokenizer")
    # Context
    context_session = ort.InferenceSession(
        "/root/autodl-tmp/models/context/context.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    context_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/context/tokenizer")
    logger_embed.info("All models loaded successfully.")

# ==================== FastAPI 应用 ====================
app = FastAPI(title="Merged ORT Service")

@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(load_models)

# ==================== 请求/响应模型 ====================
class EmbeddingRequest(BaseModel):
    input: Optional[List[str]] = None

class RerankerRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]
    sorted_indices: List[int]

class ContextRequest(BaseModel):
    text_a: str = Field(..., description="对话历史或第一段文本")
    text_b: str = Field(..., description="当前用户消息")

class ContextResponse(BaseModel):
    label: str
    label_id: int
    probability: float

CONTEXT_LABEL = ["COMPLETE", "NEED_CONTEXT"]
ID2LABEL = {i: label for i, label in enumerate(CONTEXT_LABEL)}
MAX_BATCH_SIZE = 128

# ==================== Embedding 端点 ====================
@app.post("/v1/embeddings")
async def embed(req: EmbeddingRequest):
    if not req.input:
        return {"embeddings": []}
    try:
        enc = await asyncio.to_thread(
            lambda doc: emb_tokenizer(doc, return_tensors="np", max_length=512,padding="max_length", truncation=True),
            req.input
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        embedding = await asyncio.to_thread(
            emb_session.run, None,
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        logger_embed.info(f"Embedded {len(req.input)} texts")
        return {"data": embedding[0].tolist()}   # 格式保持不变（不改6）
    except Exception as e:
        logger_embed.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding service error")

# ==================== Reranker 端点 ====================
def compute_scores_sync(query: str, documents: List[str]) -> List[float]:
    all_scores = []
    for i in range(0, len(documents), MAX_BATCH_SIZE):
        batch = documents[i:i + MAX_BATCH_SIZE]
        pairs = [[query, doc] for doc in batch]
        enc = rerank_tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True, max_length=512, padding="max_length", return_tensors="np"
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        logits = rerank_session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask
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
        logger_rerank.info(f"Reranked {len(req.documents)} documents")
        return RerankResponse(scores=scores, sorted_indices=sorted_indices)
    except Exception as e:
        logger_rerank.error(f"Reranking error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reranking service error")

# ==================== Context 端点 ====================
@app.post("/predict/context", response_model=ContextResponse)
async def predict_context(req: ContextRequest):
    try:
        enc = await asyncio.to_thread(
            lambda text_a, text_b: context_tokenizer(text_a, text_b, return_tensors="np", max_length=512,padding="max_length", truncation=True),
            req.text_a,
            req.text_b
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = await asyncio.to_thread(context_session.run, None, onnx_inputs)

        logits = outputs[0]
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        pred_id = int(np.argmax(probs, axis=-1)[0])
        result = ContextResponse(
            label=ID2LABEL[pred_id],
            label_id=pred_id,
            probability=float(probs[0, pred_id])
        )
        logger_context.info(f"Context prediction: {result.label} ({result.probability:.4f})")
        return result
    except Exception as e:
        logger_context.error(f"Context prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Context prediction error")

# ==================== 健康检查 ====================
@app.get("/health")
async def health():
    return {"status": "ok"}