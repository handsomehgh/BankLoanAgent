import json
import os
import argparse
from functools import partial
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

#bge-small-zh-v1.5

# AMP 兼容：torch>=2.3 推荐 torch.amp，旧版本回退 torch.cuda.amp
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW = False


def amp_context(device: torch.device):
    enabled = device.type == "cuda"
    if _AMP_NEW:
        return autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    return autocast(enabled=enabled, dtype=torch.float16)


def make_scaler(device: torch.device):
    enabled = device.type == "cuda"
    if _AMP_NEW:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


class EmbeddingDataSet(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
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

    # 动态 padding：query/短文档不再补齐到 max_length，显著减少无效算力
    anchor_enc = tokenizer(
        anchors,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    doc_enc = tokenizer(
        doc_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
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
    # 屏蔽批内假负例：作为其他 anchor 正例的文档列不进入 softmax 分母，
    # 避免共享正例（如 KK 术语正例被多个 anchor 复用）被误当负例推开
    num_anchors = positive_indices.size(0)
    is_pos_col = torch.zeros(sim.size(1), dtype=torch.bool, device=sim.device)
    is_pos_col[positive_indices] = True
    mask = is_pos_col.unsqueeze(0).expand(num_anchors, sim.size(1)).clone()
    mask[torch.arange(num_anchors, device=sim.device), positive_indices] = False
    if mask.any():
        sim = sim.masked_fill(mask, float("-inf"))
    return nn.CrossEntropyLoss()(sim, positive_indices)


@torch.no_grad()
def encode_texts(model, texts: List[str], tokenizer, batch_size, max_length, device):
    model.eval()
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
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

    # 将嵌入移至 CPU 计算相似度，避免 GPU 显存溢出
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


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_mrr, patience):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_mrr": best_mrr,
        "patience": patience,
    }, path)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EmbeddingModel(args.model_name).to(device)

    train_dataset = EmbeddingDataSet(args.train_file)
    val_dataset = EmbeddingDataSet(args.val_file)
    print(f"训练集 {len(train_dataset)} 条样本，验证集 {len(val_dataset)} 条样本")

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
    scaler = make_scaler(device)

    best_mrr = 0.0
    patience = 0
    start_epoch = 1
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pt")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_mrr = ckpt["best_mrr"]
        patience = ckpt["patience"]
        print(f"已从 {args.resume} 恢复训练：epoch={start_epoch}, best_mrr={best_mrr:.4f}")

    train_log = []
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        step = -1
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            doc_ids = batch["doc_input_ids"].to(device)
            doc_mask = batch["doc_attention_mask"].to(device)
            pos_idx = batch["positive_indices"].to(device)

            with amp_context(device):
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
        if step >= 0 and (step + 1) % args.gradient_accumulation_steps != 0:
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

        train_log.append({"epoch": epoch, "train_loss": avg_loss, "val_metrics": metrics})
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(train_log, f, ensure_ascii=False, indent=2)

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

        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_mrr, patience)

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
    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint path to resume training，如 output_dir/latest_checkpoint.pt（不传则从头训）")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Per GPU batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation encoding")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for InfoNCE loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--eval_top_k", type=int, nargs="+", default=[1, 5, 10], help="Top-k for recall evaluation")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
