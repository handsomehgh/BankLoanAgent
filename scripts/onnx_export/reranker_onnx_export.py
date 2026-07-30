from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = r"D:\code\pycode\BankLoanAgent\models\reranker"
ONNX_PATH = r"D:\code\pycode\BankLoanAgent\models\onnx\reranker\reranker.onnx"
Path(ONNX_PATH).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

query = "那如果我的房贷审批通过了，因为金额超过30万，放款的时候钱是不是直接打给开发商，不会进我自己的银行卡？"
doc = "一、个人住房贷款 > 还款方式 | 一手房按揭贷款提供多种还款方式,借款人可根据自身现金流情况选择。"
enc = tokenizer(query, doc, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

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

max_diff = np.max(np.abs(onnx_logits - pt_logits))
print(f"Max logit difference: {max_diff:.6e}")

onnx_probs = 1 / (1 + np.exp(-onnx_logits))
pt_probs = torch.sigmoid(torch.tensor(pt_logits)).numpy()

prob_diff = np.max(np.abs(onnx_probs - pt_probs))
print(f"Max probability difference: {prob_diff:.6e}")

print(f"ONNX score: {onnx_probs.flatten()[0]:.4f}")
print(f"PyTorch score: {pt_probs.flatten()[0]:.4f}")

if max_diff < 1e-5 and prob_diff < 1e-5:
    print("✅ ONNX export validated successfully. Output is identical to PyTorch.")
else:
    print("⚠️  Significant numerical difference detected! Check export settings.")

print(f"ONNX model saved to {ONNX_PATH}")
