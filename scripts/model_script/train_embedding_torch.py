import argparse
import json
import os
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument(name="--model_name", required=True, type=str, default="models-BAAI-small-zh-v1.5")
parser.add_argument("--train_file", type=str, default="train.jsonl")
parser.add_argument("--val_file", type=str, default="val.jsonl")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--temperature", type=float, default=0.05)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="./embedding_model")
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fp16", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--eval_top_k", type=int, nargs='+', default=[1, 3, 5, 10])
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class EmbeddingDataset(Dataset):
    def __init__(self, file_path: str):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
        return_tensors="pt"
    )
    doc_enc = tokenizer(
        doc_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    return {
        "anchor_input_ids": anchor_enc["input_ids"],
        "attention_mask": anchor_enc["attention_mask"],
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
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_emb / sum_mask
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
    return torch.cat(all_emb, dim=0)


def evaluate_retrieval(model, dataset: EmbeddingDataset, tokenizer, max_length, eval_batch_size, device, top_k):
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

    q_embs = encode_texts(model, query_texts, tokenizer, eval_batch_size, max_length=max_length, device=device)
    d_embs = encode_texts(model, id_to_doc, tokenizer, eval_batch_size, max_length=max_length, device=device)

    sim = torch.matmul(q_embs, d_embs.T).numpy()

    metrics = {f"recall@{k}": [] for k in top_k}
    metrics["mrr"] = []
    ndcg_10 = [] if 10 in top_k else None

    for qid in range(len(query_texts)):
        ranking = np.argsort(-sim[qid])
        pos_docid = qid_to_pos_docid[qid]
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


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EmbeddingModel(args.model_name)

    train_dateset = EmbeddingDataset(args.train_file)
    val_dataset = EmbeddingDataset(args.val_file)

    train_loader = DataLoader(
        train_dateset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(batch=b, tokenizer=tokenizer, max_length=args.max_length),
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_mrr = 0.0
    patience = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            anchor_ids = batch["anchor_input_ids"].to(DEVICE)
            anchor_mask = batch["anchor_attention_mask"].to(DEVICE)
            doc_ids = batch["doc_input_ids"].to(DEVICE)
            doc_mask = batch["doc_attention_mask"].to(DEVICE)
            pos_idx = batch["positive_indices"].to(DEVICE)

            with autocast(enabled=args.fp16):
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
            # 完整保存模型、配置、分词器
            model.bert.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            # 同时保存自定义的 EmbeddingModel 权重（或直接把 BERT 权重当做主权重，加载时重建）
            torch.save(model.state_dict(), os.path.join(args.output_dir, "embedding_model.pt"))
            print(f"  => saved best model (MRR {best_mrr:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 最终评估
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "embedding_model.pt"), map_location=DEVICE))
    final_metrics = evaluate_retrieval(
        model, val_dataset, tokenizer, args.max_length, args.eval_batch_size, DEVICE, args.eval_top_k
    )
    print("\nFinal Results:", ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items()))


if __name__ == "__main__":
    main()
