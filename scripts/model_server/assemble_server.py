# author hgh
# version 3.0
# merged_server.py
import asyncio
import os
import time
import logging
from typing import List, Union
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.special import expit
from transformers import AutoTokenizer

# ==================== 配置 ====================
MODEL_BASE = os.getenv("MODEL_BASE_PATH", "/root/autodl-tmp/models")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
LOG_DIR = os.getenv("LOG_DIR", "/root/autodl-tmp/logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ==================== 日志配置 ====================
def _setup_logger(name: str, filename: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(LOG_DIR, filename))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.propagate = False
    return logger


logger_embed = _setup_logger("bge_custom_embedder", "bge_custom_embedder.log")
bge_logger_embed = _setup_logger("bge_official_embedder", "bge_official_embedder.log")
logger_rerank = _setup_logger("reranker", "reranker.log")
logger_context = _setup_logger("context", "context.log")

# ==================== 全局模型变量 ====================
emb_session = None
emb_tokenizer = None
bge_emb_session = None
bge_emb_tokenizer = None
rerank_session = None
rerank_tokenizer = None
context_session = None
context_tokenizer = None


def load_models():
    global emb_session, emb_tokenizer, bge_emb_session, bge_emb_tokenizer, rerank_session, rerank_tokenizer, context_session, context_tokenizer
    # Embedder (微调模型)
    emb_session = ort.InferenceSession(
        os.path.join(MODEL_BASE, "bge_custom_embedder", "bge_embedder.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    emb_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_BASE, "bge_custom_embedder", "tokenizer"))

    # BGE Embedder (官方模型)
    bge_emb_session = ort.InferenceSession(
        os.path.join(MODEL_BASE, "bge_official_embedder", "onnx", "bge_official_embedder.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    bge_emb_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_BASE, "bge_official_embedder","tokenizer"))

    # Reranker
    rerank_session = ort.InferenceSession(
        os.path.join(MODEL_BASE, "reranker", "reranker.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    rerank_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_BASE, "reranker", "tokenizer"))

    # Context
    context_session = ort.InferenceSession(
        os.path.join(MODEL_BASE, "context", "context.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    context_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_BASE, "context", "tokenizer"))

    logger_embed.info("All models loaded successfully.")


# ==================== FastAPI 应用 ====================
app = FastAPI(title="Merged ORT Service")


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(load_models)


# ==================== 请求/响应模型 ====================
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


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


# ==================== 通用工具函数 ====================
def prepare_texts(text_input: Union[str, List[str]]) -> List[str]:
    """将输入统一为列表，并过滤空字符串"""
    if isinstance(text_input, str):
        texts = [text_input]
    else:
        texts = text_input
    return [t if t.strip() else " " for t in texts]


def encode_batch_finetuned(texts: List[str]) -> List[List[float]]:
    """微调模型编码（Mean Pooling，已内置归一化）"""
    all_embeddings = []
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i:i + MAX_BATCH_SIZE]
        enc = emb_tokenizer(
            batch,
            return_tensors="np",
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True
        )
        outputs = emb_session.run(None, {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        })
        # 微调模型输出已经是 (batch, dim) 的归一化向量
        all_embeddings.extend(outputs[0].tolist())
    return all_embeddings


def encode_batch_official(texts: List[str]) -> List[List[float]]:
    """官方 BGE 模型编码（CLS Pooling，手动归一化）"""
    all_embeddings = []
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i:i + MAX_BATCH_SIZE]
        enc = bge_emb_tokenizer(
            batch,
            return_tensors="np",
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True
        )
        outputs = bge_emb_session.run(None, {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        })
        all_embeddings.extend(outputs[0].tolist())
    return all_embeddings


# ==================== Embedding 端点（微调模型） ====================
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest):
    texts = prepare_texts(req.input)
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    start_time = time.time()
    try:
        embeddings = await asyncio.to_thread(encode_batch_finetuned, texts)
        dim = len(embeddings[0]) if embeddings else 0

        data = [
            EmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ]

        # 粗略估计 token 用量
        total_tokens = sum(len(t) for t in texts)
        usage = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}

        logger_embed.info(f"Finetuned embed: {len(texts)} texts, dim={dim}, took {time.time() - start_time:.3f}s")
        return EmbeddingResponse(
            object="list",
            data=data,
            model="bge-finetuned",
            usage=usage
        )
    except Exception as e:
        logger_embed.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding service error")


# ==================== BGE Embedding 端点（官方模型） ====================
@app.post("/v1/bge/embedding", response_model=EmbeddingResponse)
async def bge_embedding(req: EmbeddingRequest):
    texts = prepare_texts(req.input)
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    start_time = time.time()
    try:
        embeddings = await asyncio.to_thread(encode_batch_official, texts)
        dim = len(embeddings[0]) if embeddings else 0

        data = [
            EmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ]

        total_tokens = sum(len(t) for t in texts)
        usage = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}

        bge_logger_embed.info(
            f"Official BGE embed: {len(texts)} texts, dim={dim}, took {time.time() - start_time:.3f}s")
        return EmbeddingResponse(
            object="list",
            data=data,
            model="bge-official",
            usage=usage
        )
    except Exception as e:
        bge_logger_embed.error(f"BGE Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="BGE Embedding service error")


# ==================== Reranker 端点（不变） ====================
def compute_scores_sync(query: str, documents: List[str]) -> List[float]:
    all_scores = []
    for i in range(0, len(documents), MAX_BATCH_SIZE):
        batch = documents[i:i + MAX_BATCH_SIZE]
        pairs = [[query, doc] for doc in batch]
        enc = rerank_tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="np"
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        logits = rerank_session.run(None, {
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
    start_time = time.time()
    try:
        scores = await asyncio.to_thread(compute_scores_sync, req.query, req.documents)
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        logger_rerank.info(f"Reranked {len(req.documents)} docs, took {time.time() - start_time:.3f}s")
        return RerankResponse(scores=scores, sorted_indices=sorted_indices)
    except Exception as e:
        logger_rerank.error(f"Reranking error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reranking service error")


# ==================== Context 端点（不变） ====================
@app.post("/predict/context", response_model=ContextResponse)
async def predict_context(req: ContextRequest):
    start_time = time.time()
    try:
        enc = await asyncio.to_thread(
            context_tokenizer,
            req.text_a,
            req.text_b,
            return_tensors="np",
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        outputs = await asyncio.to_thread(
            context_session.run,
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        logits = outputs[0]
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        pred_id = int(np.argmax(probs, axis=-1)[0])
        result = ContextResponse(
            label=ID2LABEL[pred_id],
            label_id=pred_id,
            probability=float(probs[0, pred_id])
        )
        logger_context.info(
            f"Context prediction: {result.label} ({result.probability:.4f}), took {time.time() - start_time:.3f}s")
        return result
    except Exception as e:
        logger_context.error(f"Context prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Context prediction error")


# ==================== 健康检查 ====================
@app.get("/health")
async def health():
    return {"status": "ok"}