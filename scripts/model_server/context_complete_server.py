# author hgh
# version 1.0
import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retrieval context complete server", version="1.0")

ID2LABEL = {
    0: "COMPLETE",
    1: "NEED_CONTEXT"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("../../models/context_classifier_model")
model = AutoModelForSequenceClassification.from_pretrained("../../models/context_classifier_model").to(device)
model.eval()


class PredictRequest(BaseModel):
    text_a: str = Field(..., description="对话历史或第一段文本")
    text_b: str = Field(..., description="当前用户消息")


class PredictResponse(BaseModel):
    label: str
    label_id: int
    probability: float


@app.post("/predict/context", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        inputs = tokenizer(
            request.text_a,
            request.text_b,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            pred_prob = probs[0][pred_id].item()
        return PredictResponse(label=ID2LABEL[pred_id], label_id=pred_id, probability=round(pred_prob,4))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
