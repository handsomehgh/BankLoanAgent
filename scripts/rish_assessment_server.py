# author hgh
# version 1.0
import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Risk Assessment Classifier", version="1.0")

ID2LABEL = {
    0: "DIRECT_REPLY",
    1: "CLARIFY",
    2: "risk_assessment_skill",
    3: "calculate_dti",
    4: "calculate_ltv",
    5: "calculate_dscr",
    6: "estimate_credit_score",
    7: "query_regulation",
    8: "general_search_knowledge"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("../models/risk_assessment_classifier_model")
model = AutoModelForSequenceClassification.from_pretrained("../models/risk_assessment_classifier_model",local_files_only=True).to(device)
model.eval()

class PredictRequest(BaseModel):
    text_a: str = Field(..., description="对话历史或第一段文本")
    text_b: str = Field(..., description="当前用户消息")

class PredictResponse(BaseModel):
    label: str
    label_id: int

@app.post("/predict/risk", response_model=PredictResponse)
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
            pred_id = torch.argmax(logits, dim=-1).item()
        return PredictResponse(label=ID2LABEL[pred_id], label_id=pred_id)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}



