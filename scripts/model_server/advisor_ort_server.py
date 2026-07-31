import asyncio
import logging

import numpy as np
import onnxruntime
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
ID2LABEL = {
    i: label for i, label in enumerate(LABELS)
}
ONNX_MODEL_PATH = r"D:\code\pycode\BankLoanAgent\models\onnx\loan_advisor_bert\advisor.onnx"
ONNX_TOKENIZER_PATH = r"D:\code\pycode\BankLoanAgent\models\onnx\loan_advisor_bert\tokenizer"

session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(ONNX_TOKENIZER_PATH)

app = FastAPI(title="Advisor ORT Server")


class PredictRequest(BaseModel):
    text_a: str = Field(..., description="Text input text")
    text_b: str = Field(..., description="Text input text")


class PredictResponse(BaseModel):
    label: str
    label_id: int
    probability: float


@app.post("/predict/advisor", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        encoded = await asyncio.to_thread(
            lambda a, b: tokenizer(a, b, return_tensors="np", max_length=512, padding="max_length", truncation=True),
            request.text_a,
            request.text_b,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = await asyncio.to_thread(session.run, None, onnx_inputs)

        logits = outputs[0]
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        pred_id = int(np.argmax(probs, axis=-1)[0])
        return PredictResponse(
            label=ID2LABEL[pred_id],
            label_id=pred_id,
            probability=float(probs[0,pred_id]),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
