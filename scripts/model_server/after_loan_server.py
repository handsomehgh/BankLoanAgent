# author hgh
# version 1.0
import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="After Loan Classifier", version="1.0")

ID2LABEL = {
    0: "DIRECT_REPLY",
    1: "CLARIFY",
    2: "prepayment_evaluation_skill",
    3: "extension_management_skill",
    4: "overdue_handling_skill",
    5: "repayment_method_switch_skill",
    6: "calculate_prepayment",
    7: "check_extension_eligibility",
    8: "calculate_extension_plan",
    9: "calculate_overdue_penalty",
    10: "calculate_repayment_method_switch",
    11: "generate_repayment_schedule",
    12: "generate_settlement_certificate",
    13: "general_search_knowledge"
  }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("../../models/after_loan_classifier_model")
model = AutoModelForSequenceClassification.from_pretrained("../../models/after_loan_classifier_model", local_files_only=True).to(device)
model.eval()

class PredictRequest(BaseModel):
    text_a: str = Field(..., description="对话历史或第一段文本")
    text_b: str = Field(..., description="当前用户消息")

class PredictResponse(BaseModel):
    label: str
    label_id: int
    probability: float

@app.post("/predict/after", response_model=PredictResponse)
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



