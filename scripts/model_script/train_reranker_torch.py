import argparse
import json
import math
import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

#bge-reranker-v2-m3

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


class RerankerDataSet(Dataset):
    def __init__(self, file_path: str):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                query = item.get("query", "")
                doc = item.get("document", "")
                label = item.get("label", None)
                # 注意：label=0 是合法负样本，不能用真值判断过滤
                if not query or not doc or label is None:
                    continue
                self.samples.append(
                    {
                        "query": query,
                        "document": doc,
                        "label": int(label)
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length):
    query_list = [item["query"] for item in batch]
    doc_list = [item["document"] for item in batch]
    label_list = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    encoded = tokenizer(query_list, doc_list, max_length=max_length, padding=True, truncation="longest_first",
                        return_tensors="pt")

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": label_list
    }


def load_model_and_tokenizer(model_name, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            print("Added [PAD] token and resized model embeddings.")

    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device), tokenizer


def compute_metrics(all_labels, all_probs):
    labels = np.array(all_labels).astype(int)
    preds = (np.array(all_probs) > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_positive = f1_score(labels, preds, pos_label=1)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, all_probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_positive": f1_positive,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }


def train_epoch(model, loader: DataLoader, optimizer, scheduler, scaler, criterion, max_grad_norm,
                gradient_accumulation_steps, device):
    model.train()
    all_labels, all_probs = [], []
    total_loss = 0.0

    optimizer.zero_grad()
    step = -1
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with amp_context(device):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)
            loss = criterion(logits, labels)

        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss * gradient_accumulation_steps
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    if step >= 0 and (step + 1) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with amp_context(device):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def evaluate_ranking(model, rank_file: str, tokenizer, max_length, batch_size, device, top_ks=(1, 3)):
    """listwise 排序评估：消费 val_rank.jsonl，按 query 分组对候选打分，计算 MRR / HitRate@k。
    对齐线上 reranker 的真实任务（融合候选精排），作为主选模指标。"""
    groups = []
    with open(rank_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("candidates"):
                groups.append(item)
    if not groups:
        return {}

    pairs, group_slices, gold_indices = [], [], []
    start = 0
    for item in groups:
        candidates = item["candidates"]
        for cand in candidates:
            pairs.append((item["query"], cand["document"]))
        gold = next(
            (i for i, cand in enumerate(candidates) if cand.get("chunk_id") == item.get("positive_chunk_id")),
            None
        )
        if gold is None:
            continue
        gold_indices.append(gold)
        group_slices.append((start, start + len(candidates)))
        start += len(candidates)

    model.eval()
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        q_list = [p[0] for p in batch_pairs]
        d_list = [p[1] for p in batch_pairs]
        encoded = tokenizer(q_list, d_list, max_length=max_length, padding=True,
                            truncation="longest_first", return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with amp_context(device):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())

    scores_arr = np.array(scores)
    mrr, hits = [], {k: [] for k in top_ks}
    for (s, e), gold in zip(group_slices, gold_indices):
        order = np.argsort(-scores_arr[s:e])
        rank = int(np.where(order == gold)[0][0]) + 1
        mrr.append(1.0 / rank)
        for k in top_ks:
            hits[k].append(1.0 if rank <= k else 0.0)

    result = {"mrr": float(np.mean(mrr))}
    for k in top_ks:
        result[f"hit@{k}"] = float(np.mean(hits[k]))
    return result


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, scaler, epoch, best_metric, patience_counter):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
    }, path)
    tokenizer.save_pretrained(os.path.join(os.path.dirname(path), "last_tokenizer"))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    train_dataset = RerankerDataSet(args.train_file)
    val_dataset = RerankerDataSet(args.val_file)
    print(f"训练集 {len(train_dataset)} 条（正样本 "
          f"{sum(1 for s in train_dataset.samples if s['label'] == 1)}），验证集 {len(val_dataset)} 条")

    train_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    val_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=val_collate,
                            num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = make_scaler(device)
    criterion = nn.BCEWithLogitsLoss()

    best_metric = 0.0
    patience_counter = 0
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
        best_metric = ckpt["best_metric"]
        patience_counter = ckpt["patience_counter"]
        print(f"已从 {args.resume} 恢复训练：epoch={start_epoch}, best={best_metric:.4f}")

    use_rank_metric = bool(args.val_rank_file) and os.path.exists(args.val_rank_file)
    metric_name = "MRR" if use_rank_metric else "AUC"
    if use_rank_metric:
        print(f"选模指标：val_rank MRR（辅助 HitRate），文件 {args.val_rank_file}")
    else:
        print("选模指标：pointwise AUC（未提供 --val_rank_file）")

    train_log = []
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            args.max_grad_norm, args.gradient_accumulation_steps, device
        )
        print(f"Train Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1(macro): {train_metrics['f1_macro']:.4f} | "
              f"F1(pos): {train_metrics['f1_positive']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f} | "
              f"AUC: {train_metrics['auc']:.4f}")

        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        print(f"Val   Loss: {val_loss:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1(macro): {val_metrics['f1_macro']:.4f} | "
              f"F1(pos): {val_metrics['f1_positive']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f}")

        rank_metrics = {}
        if use_rank_metric:
            rank_metrics = evaluate_ranking(
                model, args.val_rank_file, tokenizer, args.max_length,
                args.eval_batch_size, device
            )
            print("Val   Rank:", ", ".join(f"{k}={v:.4f}" for k, v in rank_metrics.items()))

        current_metric = rank_metrics.get("mrr", 0.0) if use_rank_metric else val_metrics["auc"]
        train_log.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "rank_metrics": rank_metrics, "select_metric": current_metric,
        })
        with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(train_log, f, ensure_ascii=False, indent=2)

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"  => saved best model ({metric_name} {best_metric:.4f})")
        else:
            patience_counter += 1

        save_checkpoint(checkpoint_path, model, tokenizer, optimizer, scheduler, scaler,
                        epoch, best_metric, patience_counter)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(args.output_dir, "best_model")
    ).to(device)
    _, final_metrics = eval_epoch(model, val_loader, criterion, device)
    print(f"\nFinal Results -> "
          f"Acc: {final_metrics['accuracy']:.4f} | "
          f"F1(macro): {final_metrics['f1_macro']:.4f} | "
          f"F1(pos): {final_metrics['f1_positive']:.4f} | "
          f"Precision: {final_metrics['precision']:.4f} | "
          f"Recall: {final_metrics['recall']:.4f} | "
          f"AUC: {final_metrics['auc']:.4f}")
    if use_rank_metric:
        final_rank = evaluate_ranking(
            model, args.val_rank_file, tokenizer, args.max_length, args.eval_batch_size, device
        )
        print("Final Rank   ->", ", ".join(f"{k}={v:.4f}" for k, v in final_rank.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reranker fine-tuning with Cross-Encoder")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-base",
                        help="Pretrained model name or path (注意：线上 reranker 配置基座为 "
                             "BAAI/bge-reranker-v2-m3，训练基座需与部署目标保持一致)")
    parser.add_argument("--train_file", type=str, default="./train_reranker.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--val_file", type=str, default="./val_reranker.jsonl",
                        help="Path to validation JSONL file")
    parser.add_argument("--val_rank_file", type=str, default="./val_rank.jsonl",
                        help="Path to listwise validation file (MRR/HitRate 选模指标，文件不存在时退回 AUC)")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/models/torch_reranker",
                        help="Directory to save model")
    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint path to resume training，如 output_dir/latest_checkpoint.pt（不传则从头训）")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
