# author hgh
# version 1.0
import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loan Advisor Classifier", version="1.0")

ID2LABEL = {
    0: "DIRECT_REPLY",
    1: "CLARIFY",
    2: "apply_home_loan_skill",
    3: "apply_consumer_loan_skill",
    4: "calculate_monthly_payment",
    5: "query_interest_rate",
    6: "calculate_loan_total_cost",
    7: "calculate_max_loan_amount",
    8: "check_loan_eligibility",
    9: "compare_loan_products",
    10: "generate_repayment_schedule",
    11: "general_search_knowledge",
    12: "query_loan_interest",
    13: "upsert_loan_interest",
    14: "urge_loan_interest"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("../../models/loan_advisor_classifier_model")
model = AutoModelForSequenceClassification.from_pretrained("../../models/loan_advisor_classifier_model", local_files_only=True).to(device)
model.eval()

class PredictRequest(BaseModel):
    text_a: str = Field(..., description="对话历史或第一段文本")
    text_b: str = Field(..., description="当前用户消息")

class PredictResponse(BaseModel):
    label: str
    label_id: int
    probability: float

@app.post("/predict/advisor", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        inputs = tokenizer(
            request.text_a[:500] if request.text_a else "",
            request.text_b,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_prob = probs[0][pred_id].item()
        return PredictResponse(label=ID2LABEL[pred_id], label_id=pred_id,probability=pred_prob)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}



