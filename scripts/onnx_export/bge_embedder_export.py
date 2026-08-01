# author hgh
# version 4.1 (修复重复输出名称问题)
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoTokenizer
import onnxruntime as ort

class WrappedEmbedder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        embedding = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})['sentence_embedding']
        return embedding

# ---------- 配置 ----------
model_name = "BAAI/bge-base-zh-v1.5"
onnx_dir = r"D:\code\pycode\BankLoanAgent\models\onnx\bge_official_embedder"
onnx_path = os.path.join(onnx_dir, "bge_official_embedder.onnx")
os.makedirs(onnx_dir, exist_ok=True)

# ---------- 准备模型 ----------
model = WrappedEmbedder(model_name)
model.eval()
# 使用新方法名（避免弃用警告）
print(f"模型输出维度: {model.model.get_embedding_dimension()}")

# ---------- 准备 dummy 输入 ----------
tokenizer = model.tokenizer
dummy_texts = ["测试文本", "这是另一个测试文本"]
dummy = tokenizer(
    dummy_texts,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)
input_ids = dummy['input_ids']
attention_mask = dummy['attention_mask']

print(f"📌 输入 shape:")
print(f"   input_ids:      {input_ids.shape}")
print(f"   attention_mask: {attention_mask.shape}")

# ---------- 导出 ONNX ----------
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["custom_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=18,
        do_constant_folding=True
    )
model.tokenizer.save_pretrained(onnx_dir)

print(f"✅ ONNX 模型已导出至: {onnx_path}")

# ---------- 验证 ----------
with torch.no_grad():
    pt_output = model(input_ids, attention_mask).cpu().numpy()

ort_session = ort.InferenceSession(onnx_path)
ort_inputs = {
    'input_ids': input_ids.cpu().numpy(),
    'attention_mask': attention_mask.cpu().numpy()
}
ort_output = ort_session.run(['custom_embedding'], ort_inputs)[0]

max_diff = np.max(np.abs(pt_output - ort_output))
print(f"\n📊 输出验证:")
print(f"   PyTorch 输出 shape: {pt_output.shape}")
print(f"   ONNX 输出 shape:    {ort_output.shape}")
print(f"   最大绝对差异:       {max_diff:.6e}")

if max_diff < 1e-5:
    print("   ✅ 导出成功，ONNX 输出与 PyTorch 完全一致！")
else:
    print("   ⚠️ 存在较大差异，请检查模型导出过程。")