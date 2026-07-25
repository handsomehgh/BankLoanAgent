# author hgh
# version 1.0
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS = [
    "DIRECT_REPLY",
    "CLARIFY",
    "apply_home_loan_skill",
    "apply_consumer_loan_skill",
    "calculate_monthly_payment",
    "query_interest_rate",
    "calculate_loan_total_cost",
    "calculate_max_loan_amount",
    "check_loan_eligibility",
    "compare_loan_products",
    "generate_repayment_schedule",
    "general_search_knowledge",
    "query_loan_interest",
    "upsert_loan_interest",
    "urge_loan_interest",
]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "onnx" / "advisor_bert"
ONNX_PATH = os.path.join(MODEL_DIR, "advisor.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer")

session = ort.InferenceSession(ONNX_PATH,providers=["CUDAExecutionProvider""CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

app = FastAPI(title="Advisor ORT")


class PredictRequest(BaseModel):
    text_a: str = Field(..., description="对话历史或第一段文本")
    text_b: str = Field(..., description="当前用户消息")


class PredictResponse(BaseModel):
    label: str
    label_id: int
    probability: float


@app.post("/predict/advisor", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        encoded = tokenizer(
            request.text_a,
            request.text_b,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="np"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        logits = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })[0][0]

        probs = softmax(logits)
        pred_id = int(np.argmax(probs))
        probs = float(probs[pred_id])
        return PredictResponse(
            label=ID2LABEL[pred_id],
            label_id=pred_id,
            probability=probs,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
