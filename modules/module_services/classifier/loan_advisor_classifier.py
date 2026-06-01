# author hgh
# version 1.0
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# 12 类标签，顺序必须与训练时一致
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
}

class LoanAdvisorClassifier:
    def __init__(self,model_path: str,device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading loan advisor BERT classifier device:  {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True).to(self.device)
        self.model.eval()
        logger.info("Loading loan advisor BERT classifier completed")

    def predict(self,text_a: str,text_b: str) -> str:
        inputs = self.tokenizer(
            text_a[:500] if text_a else "",
            text_b,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_id  = torch.argmax(logits, dim=-1).item()

        label = ID2LABEL[predicted_id]
        logger.debug(f"BERT predict: '{text_b[:50]}...' -> {label}")
        return label
