from pathlib import Path

import numpy as np
import onnx
import torch
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "D:\code\pycode\BankLoanAgent\models\loan_advisor_classifier_model"
ONNX_PATH = "D:\code\pycode\BankLoanAgent\models\onnx\loan_advisor_bert"
Path(ONNX_PATH).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval().to(device)

text_a = "最近对话：用户: 缩短期限和减少月供哪个划算？"
text_b = "提前还款的话，违约金怎么算？"
encoded = tokenizer(text_a, text_b, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

dynamic_shapes = {
    "input_ids": {0: "batch_size", 1: "seq_length"},
    "attention_mask": {0: "batch_size", 1: "seq_length"}
}

onnx_file = str(ONNX_PATH) / "advisor.onnx"

with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        do_constant_folding=True
    )

onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model structure valid.")

session = ort.InferenceSession(onnx_file)
onnx_logits = session.run(None, {
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy()
})[0]

with torch.no_grad():
    pt_logits = model(input_ids, attention_mask).logits.cpu().numpy()

max_diff = np.max(np.abs(onnx_logits - pt_logits))
print(f"Max logit difference: {max_diff:.6e}")
if max_diff < 1e-5:
    print("✅ ONNX export validated successfully.")
else:
    print("⚠️ Significant numerical difference detected!")

print(f"ONNX model saved to {onnx_file}")
