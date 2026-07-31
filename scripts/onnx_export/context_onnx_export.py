# author hgh
# version 1.0
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = r"D:\code\pycode\BankLoanAgent\models\context_classifier_model"
ONNX_PATH = r"D:\code\pycode\BankLoanAgent\models\onnx\context\context.onnx"
Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

text_a = "This is a test"
text_b = "This is a test"

enc = tokenizer(text_a,text_b,return_tensors="pt",max_length=512,padding="max_length",truncation=True)
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)


dynamic_shapes = {
    "input_ids": {0: "batch_size", 1: "seq_length"},
    "attention_mask": {0: "batch_size", 1: "seq_length"}
}

with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        do_constant_folding=True
    )

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model structure valid.")

input_ids_np = input_ids.cpu().numpy()
attention_mask_np = attention_mask.cpu().numpy()

session = onnxruntime.InferenceSession(ONNX_PATH)
onnx_logits = session.run(None, {"input_ids": input_ids_np, "attention_mask": attention_mask_np})[0]

with torch.no_grad():
    pt_logits = model(input_ids, attention_mask).logits.cpu().numpy()

print(f"onnx-----------{onnx_logits}")
print(f"model------------{pt_logits}")

max_diff = np.max(np.abs(onnx_logits - pt_logits))
print(f"Max logit difference: {max_diff:.6e}")
if max_diff < 1e-5:
    print("✅ ONNX export validated successfully.")
else:
    print("⚠️ Significant numerical difference detected!")