# author hgh
# version 1.0
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoTokenizer
import onnxruntime as ort

from scripts.model_script.train_embedding_torch import EmbeddingModel

PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "bge_small_loan"
ONNX_PATH = PROJECT_ROOT / "models" / "onnx"/ "bge_embedder"
ONNX_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingModel(str(MODEL_PATH)).to(DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

texts = ["这是一条测试查询"]
enc = tokenizer(texts,truncation=True,max_length=512,padding="max_length",return_tensors="pt")
input_ids = enc["input_ids"].to(DEVICE)
attention_mask = enc["attention_mask"].to(DEVICE)

dynamic_shapes = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
}

with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids,attention_mask),
        ONNX_PATH / "bge_embedder.onnx",
        input_names=["input_ids","attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
print(f"✅ ONNX 模型已保存至 {ONNX_PATH}")


onnx_model = onnx.load(ONNX_PATH / "bge_embedder.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model structure valid.")

session = ort.InferenceSession(str(ONNX_PATH / "bge_embedder.onnx"))
onnx_emb = session.run(None,{
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy(),
})[0]

with torch.no_grad():
    pt_emb = model(input_ids, attention_mask).cpu().numpy()

max_diff = np.max(np.abs(onnx_emb - pt_emb))
print(f"最大嵌入差异: {max_diff:.6e}")
if max_diff < 1e-5:
    print("✅ ONNX 导出验证成功，输出与 PyTorch 完全一致。")












