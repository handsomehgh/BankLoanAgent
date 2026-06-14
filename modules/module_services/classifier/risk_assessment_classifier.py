# author hgh
# version 1.0
# import logging
# from typing import Optional
#
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# logger = logging.getLogger(__name__)
#
# ID2LABEL = {
#     0: "DIRECT_REPLY",
#     1: "CLARIFY",
#     2: "risk_assessment_skill",
#     3: "calculate_dti",
#     4: "calculate_ltv",
#     5: "calculate_dscr",
#     6: "estimate_credit_score",
#     7: "query_regulation",
#     8: "general_search_knowledge"
# }
# class RiskAssessmentClassifier:
#     def __init__(self, model_path: str,device: Optional[str] = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         logger.info(f"Using device: {self.device}")
#
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True).to(self.device)
#         self.model.eval()
#         logger.info("Loading risk assessment BERT classifier completed")
#
#     def predict(self,text_a: str,text_b: str) -> str:
#         inputs = self.tokenizer(
#             text_a[:500] if text_a else "",
#             text_b,
#             truncation=True,
#             max_length=512,
#             padding="max_length",
#             return_tensors="pt"
#         )
#
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             predicted_id = torch.argmax(logits,dim=-1).item()
#
#         label = ID2LABEL[predicted_id]
#         logger.debug(f"BERT predict: '{text_b[:50]}...' -> {label}")
#         return label
import json
import logging
import os
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class RiskAssessmentClassifier:
    def __init__(self, api_url: str = "http://localhost:8004"):
        self.api_url = api_url

    def predict(self, text_a: str, text_b: str) -> str:
        try:
            resp = requests.post(
                f"{self.api_url}/predict/risk",
                json={"text_a": text_a, "text_b": text_b},
                timeout=30
            )
            resp.raise_for_status()
            probs = resp.json()["probability"]
            if probs and probs < 0.6:
                low_data_dir = PROJECT_ROOT / "data" / "wheel" / "tool"
                os.makedirs(low_data_dir, exist_ok=True)
                low_data_file = low_data_dir / "after_loan.jsonl"
                with open(str(low_data_file), "a", encoding="utf-8") as f:
                    record = {"text_a": text_a, "text_b": text_b, "label": resp.json()["label"]}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return resp.json()["label"]
        except Exception as e:
            logging.error(f"After loan bert failed to make prediction: {e}")
            return "CLARIFY"

