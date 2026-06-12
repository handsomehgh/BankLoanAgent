#!/usr/bin/env python3
"""
Embedding 对比学习训练脚本（InfoNCE / MultipleNegativesRankingLoss）
优雅设计原则：
1. Dataset 只保存原始文本，不做 tokenize
2. collate_fn 负责批量 tokenize（利用 HuggingFace 高速后端）
3. 全局评估使用 encode_texts 统一编码，构建检索指标
4. 训练与评估复用同一个 Dataset 实例和编码函数
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
from typing import List, Dict, Tuple

# -------------------- 参数 --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-base-chinese")
parser.add_argument("--train_file", type=str, default="train.jsonl")
parser.add_argument("--val_file", type=str, default="val.jsonl")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--temperature", type=float, default=0.05)
parser.add_argument("--output_dir", type=str, default="./embedding_model")
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--eval_top_k", type=int, nargs='+', default=[1, 3, 5, 10])
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# -------------------- 1. Dataset：只存原始文本 --------------------
class EmbeddingDataset(Dataset):
    def __init__(self, file_path: str):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                anchor = item.get("anchor") or item.get("query")
                positive = item.get("positive")
                if not anchor or not positive:
                    continue
                negatives = [n for n in item.get("negatives", []) if n]
                self.samples.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negatives": negatives
                })
        print(f"Loaded {len(self.samples)} samples from {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------- 2. collate_fn：批量 tokenize --------------------
def collate_fn(batch: List[Dict], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    anchors = [item["anchor"] for item in batch]
    doc_texts = []
    doc_sample_idx = []

    for i, item in enumerate(batch):
        doc_texts.append(item["positive"])
        doc_sample_idx.append(i)
        for neg in item["negatives"]:
            doc_texts.append(neg)
            doc_sample_idx.append(i)

    anchor_enc = tokenizer(anchors, truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt")
    doc_enc = tokenizer(doc_texts, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")

    return {
        "anchor_input_ids": anchor_enc["input_ids"],
        "anchor_attention_mask": anchor_enc["attention_mask"],
        "doc_input_ids": doc_enc["input_ids"],
        "doc_attention_mask": doc_enc["attention_mask"],
        "doc_sample_idx": torch.tensor(doc_sample_idx, dtype=torch.long)
    }

# -------------------- 3. 模型：BERT + Mean Pooling + L2 Norm --------------------
class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        # mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_emb / sum_mask
        # L2 normalize
        return nn.functional.normalize(pooled, p=2, dim=1)

# -------------------- 4. 损失函数：InfoNCE --------------------
def info_nce_loss(anchor_emb, doc_emb, doc_sample_idx, temperature):
    sim = torch.matmul(anchor_emb, doc_emb.T) / temperature   # (B, total_docs)
    B = anchor_emb.size(0)
    labels = torch.zeros(B, dtype=torch.long, device=anchor_emb.device)
    for i in range(B):
        pos_idx = (doc_sample_idx == i).nonzero(as_tuple=True)[0]
        labels[i] = pos_idx[0] if len(pos_idx) > 0 else 0
    return nn.CrossEntropyLoss()(sim, labels)

# -------------------- 5. 通用编码函数（评估专用，带 no_grad）--------------------
@torch.no_grad()
def encode_texts(model, texts: List[str], tokenizer, max_length, batch_size, device):
    model.eval()
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        emb = model(input_ids, attention_mask)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)

# -------------------- 6. 评估：构建全局文档库并计算检索指标 --------------------
def evaluate_retrieval(model, dataset: EmbeddingDataset, tokenizer, max_length, eval_batch_size, device, top_k):
    # 利用 dataset.samples 构建全局索引（一次遍历，无文件读取）
    query_texts = []
    doc_to_id = {}
    id_to_doc = []
    qid_to_pos_docid = {}

    for sample in dataset.samples:
        pos = sample["positive"]
        if pos not in doc_to_id:
            doc_to_id[pos] = len(id_to_doc)
            id_to_doc.append(pos)
        for neg in sample["negatives"]:
            if neg not in doc_to_id:
                doc_to_id[neg] = len(id_to_doc)
                id_to_doc.append(neg)

    for idx, sample in enumerate(dataset.samples):
        query_texts.append(sample["anchor"])
        qid_to_pos_docid[idx] = doc_to_id[sample["positive"]]

    q_embs = encode_texts(model, query_texts, tokenizer, max_length, eval_batch_size, device)
    d_embs = encode_texts(model, id_to_doc, tokenizer, max_length, eval_batch_size, device)

    sim = torch.matmul(q_embs, d_embs.T).numpy()  # (N_q, N_d)

    metrics = {f"recall@{k}": [] for k in top_k}
    metrics["mrr"] = []
    ndcg_10 = [] if 10 in top_k else None

    for qid in range(len(query_texts)):
        ranking = np.argsort(-sim[qid])
        pos_docid = qid_to_pos_docid[qid]
        rank = int(np.where(ranking == pos_docid)[0][0]) + 1  # 1-based

        for k in top_k:
            metrics[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
        metrics["mrr"].append(1.0 / rank)
        if 10 in top_k:
            ndcg_10.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)

    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    if 10 in top_k:
        result["ndcg@10"] = float(np.mean(ndcg_10))
    return result

# -------------------- 7. 主训练流程 --------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EmbeddingModel(args.model_name).to(DEVICE)

    train_dataset = EmbeddingDataset(args.train_file)
    val_dataset = EmbeddingDataset(args.val_file)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_mrr = 0.0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            anchor_ids = batch["anchor_input_ids"].to(DEVICE)
            anchor_mask = batch["anchor_attention_mask"].to(DEVICE)
            doc_ids = batch["doc_input_ids"].to(DEVICE)
            doc_mask = batch["doc_attention_mask"].to(DEVICE)
            doc_idx = batch["doc_sample_idx"].to(DEVICE)

            optimizer.zero_grad()
            anc_emb = model(anchor_ids, anchor_mask)
            doc_emb = model(doc_ids, doc_mask)
            loss = info_nce_loss(anc_emb, doc_emb, doc_idx, args.temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        # 验证
        metrics = evaluate_retrieval(
            model, val_dataset, tokenizer, args.max_length, args.eval_batch_size, DEVICE, args.eval_top_k
        )
        print("Val Metrics:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            patience = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  => saved best model (MRR {best_mrr:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 最终评估
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    final_metrics = evaluate_retrieval(
        model, val_dataset, tokenizer, args.max_length, args.eval_batch_size, DEVICE, args.eval_top_k
    )
    print("\nFinal Results:", ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items()))

if __name__ == "__main__":
    main()