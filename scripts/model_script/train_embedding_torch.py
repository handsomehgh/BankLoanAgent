import json
import os
import argparse
from functools import partial
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class EmbeddingDataSet(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                anchor = item.get("anchor", "")
                positive = item.get("positive", "")
                if not anchor or not positive:
                    continue
                negatives = item.get("negatives", [])
                self.samples.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negatives": negatives
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch: List[Dict], tokenizer, max_length: int):
    anchors = [item["anchor"] for item in batch]
    doc_texts = []
    positive_indices = []
    idx = 0
    for item in batch:
        doc_texts.append(item["positive"])
        positive_indices.append(idx)
        idx += 1
        for neg in item["negatives"]:
            doc_texts.append(neg)
            idx += 1

    anchor_enc = tokenizer(
        anchors,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    doc_enc = tokenizer(
        doc_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    return {
        "anchor_input_ids": anchor_enc["input_ids"],
        "anchor_attention_mask": anchor_enc["attention_mask"],
        "doc_input_ids": doc_enc["input_ids"],
        "doc_attention_mask": doc_enc["attention_mask"],
        "positive_indices": torch.tensor(positive_indices, dtype=torch.long)
    }


class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out["last_hidden_state"]
        pooled = torch.sum(last_hidden * attention_mask.unsqueeze(-1), dim=1) / (
            attention_mask.sum(1).unsqueeze(-1).clamp(min=1e-9)
        )
        return nn.functional.normalize(pooled, p=2, dim=1)


def info_nce_loss(anchor_emb, doc_emb, positive_indices, temperature):
    sim = torch.matmul(anchor_emb, doc_emb.T) / temperature
    return nn.CrossEntropyLoss()(sim, positive_indices)


@torch.no_grad()
def encode_texts(model, texts: List[str], tokenizer, batch_size, max_length, device):
    model.eval()
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        emb = model(input_ids, attention_mask)
        all_emb.append(emb)
    return torch.cat(all_emb)


def evaluate_retrieval(model, dataset: EmbeddingDataSet, tokenizer, max_length, eval_batch_size, device, top_k):
    query_texts = []
    doc_to_id = {}
    id_to_doc = []
    qid_to_pos_docid = {}

    # 收集所有文档（去重）
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

    # 编码查询和文档
    q_embs = encode_texts(model, query_texts, tokenizer, eval_batch_size, max_length=max_length, device=device)
    d_embs = encode_texts(model, id_to_doc, tokenizer, eval_batch_size, max_length=max_length, device=device)

    # 修复：将嵌入移至 CPU 计算相似度，避免 GPU 显存溢出
    q_embs_cpu = q_embs.cpu()
    d_embs_cpu = d_embs.cpu()
    sim = torch.matmul(q_embs_cpu, d_embs_cpu.T).numpy()

    metrics = {f"recall@{k}": [] for k in top_k}
    metrics["mrr"] = []
    ndcg_10 = [] if 10 in top_k else None

    for qid in range(len(query_texts)):
        ranking = np.argsort(-sim[qid])
        pos_docid = qid_to_pos_docid[qid]
        # 查找正样本在排序中的位置（从1开始）
        rank = int(np.where(ranking == pos_docid)[0][0]) + 1

        for k in top_k:
            metrics[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
        metrics["mrr"].append(1.0 / rank)
        if 10 in top_k:
            ndcg_10.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)

    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    if 10 in top_k:
        result["ndcg@10"] = float(np.mean(ndcg_10))
    return result


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EmbeddingModel(args.model_name).to(device)

    train_dataset = EmbeddingDataSet(args.train_file)
    val_dataset = EmbeddingDataSet(args.val_file)
    train_collate = partial(collate_fn, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler()

    best_mrr = 0.0
    patience = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            doc_ids = batch["doc_input_ids"].to(device)
            doc_mask = batch["doc_attention_mask"].to(device)
            pos_idx = batch["positive_indices"].to(device)

            with autocast(enabled=True, dtype=torch.float16):
                anc_emb = model(anchor_ids, anchor_mask)
                doc_emb = model(doc_ids, doc_mask)
                loss = info_nce_loss(anc_emb, doc_emb, pos_idx, args.temperature)

            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps

        # 处理残余梯度（不足一个累积步数的剩余batch）
        if (step + 1) % args.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        # 验证
        metrics = evaluate_retrieval(
            model, val_dataset, tokenizer, args.max_length, args.eval_batch_size, device, args.eval_top_k
        )
        print("Val Metrics:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            patience = 0
            # 保存最佳模型（只需保存完整模型参数，便于推理时直接加载）
            torch.save(model.state_dict(), os.path.join(args.output_dir, "embedding_model.pt"))
            # 同时保存配置和分词器，便于推理
            model.bert.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  => saved best model (MRR {best_mrr:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 最终评估
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "embedding_model.pt"), map_location=device))
    final_metrics = evaluate_retrieval(
        model, val_dataset, tokenizer, args.max_length, args.eval_batch_size, device, args.eval_top_k
    )
    print("\nFinal Results:", ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a bi-encoder embedding model with InfoNCE loss")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-zh-v1.5", help="Pretrained model name or path")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save model")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Per GPU batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation encoding")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for InfoNCE loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--eval_top_k", type=int, nargs="+", default=[1, 5, 10], help="Top-k for recall evaluation")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
