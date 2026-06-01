#!/usr/bin/env python3
"""
==========================================================================
银行贷款领域 Embedding 训练数据生成脚本
严格按照商定方案：Query-Knowledge + Knowledge-Knowledge
Teacher: bge-m3 + text2vec-large-chinese + m3e-large
Student: BAAI/bge-small-zh-v1.5
==========================================================================
用法：
    python generate_train_data.py --mode qk   # 只生成 Query-Knowledge
    python generate_train_data.py --mode kk   # 只生成 Knowledge-Knowledge
    python generate_train_data.py --mode all  # 两者都生成
==========================================================================
"""

import os
import json
import random
import argparse
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================================
# 配置区
# ============================================================================
# 文件路径
CHUNK_FILE = "chunked_docs.jsonl"
OUTPUT_QK = "qk_train.jsonl"
OUTPUT_KK = "kk_train.jsonl"
OUTPUT_ALL = "train_data.jsonl"
OUTPUT_EVAL = "eval_kk_data.jsonl"

TEACHER_MODELS = [
    "BAAI/bge-m3",
    "GanymedeNil/text2vec-large-chinese",
    "moka-ai/m3e-large"
]

# 检索配置
RETRIEVAL_TOP_K = 200
HARD_NEG_CANDIDATE_SIZE = 20
JACCARD_THRESHOLD = 0.8
COSINE_THRESHOLD = 0.9

# 负例数量
QK_HARD_NEG_COUNT = 3
QK_EASY_NEG_COUNT = 2
KK_HARD_NEG_COUNT = 2
KK_EASY_NEG_COUNT = 1

# DeepSeek API 配置
LLM_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-v4-flash"

# 数据增强配置
FAQ_PARAPHRASE_COUNT = 3
MANUAL_QUERY_COUNT = 5
PROCESS_QUERY_COUNT = 3
REGULATION_QUERY_COUNT = 3
GLOSSARY_QUERY_COUNT = 3
KK_REWRITE_COUNT = 3
KK_CROSS_PAIR_COUNT = 1

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
BATCH_SIZE = 64


# ============================================================================
# 工具函数
# ============================================================================

def load_chunks(file_path: str) -> List[Dict]:
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"从 {file_path} 加载了 {len(chunks)} 个 chunk")
    return chunks


def get_content(chunk: Dict) -> str:
    return chunk.get("content", "")


def get_metadata(chunk: Dict) -> Dict:
    return chunk.get("metadata", {})


def get_chunk_id(chunk: Dict) -> str:
    meta = get_metadata(chunk)
    return meta.get("chunk_id", hashlib.md5(get_content(chunk).encode()).hexdigest())


def get_product_type(chunk: Dict) -> str:
    meta = get_metadata(chunk)
    return meta.get("product_type", "通用")


def get_topics(chunk: Dict) -> List[str]:
    meta = get_metadata(chunk)
    return meta.get("topics", [])


def get_source_type(chunk: Dict) -> str:
    meta = get_metadata(chunk)
    return meta.get("source_type", "")


def get_question(chunk: Dict) -> str:
    meta = get_metadata(chunk)
    return meta.get("question", "")


def get_answer(chunk: Dict) -> str:
    meta = get_metadata(chunk)
    return meta.get("answer", "")


def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(text1)
    set2 = set(text2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


# ============================================================================
# LLM 客户端（基于 requests，自动走代理，复用连接）
# ============================================================================

class LLMClient:
    """DeepSeek API 客户端（使用 requests.Session）"""

    def __init__(self, api_key: str, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def generate(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_reasoning": False  # DeepSeek 通用参数，关闭推理链
        }
        start = time.time()
        try:
            resp = self.session.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            elapsed = time.time() - start
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(
                f"API调用耗时: {elapsed:.1f}s | 输入token: {usage.get('prompt_tokens')} | 输出token: {usage.get('completion_tokens')}")
            if content is None:
                reasoning = data["choices"][0]["message"].get("reasoning_content", "")
                print(f"LLM 返回空 content，reasoning 预览: {reasoning[:200]}...")
                return ""
            return content.strip()
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return ""


# ============================================================================
# Prompt 模板
# ============================================================================

def build_faq_paraphrase_prompt(question: str, answer: str) -> str:
    return f"""将以下用户问题改写为 {FAQ_PARAPHRASE_COUNT} 种不同的口语化问法。

要求：
- 保持原问题的核心意图不变
- 可以换用不同的句式（反问、缩略、加语气词）
- 不要添加原问题没有的额外意图

原问题：{question}
参考答案：{answer}

请输出 {FAQ_PARAPHRASE_COUNT} 个改写后的问题，每行一个，用序号开头：
1. 
2. 
..."""


def build_manual_query_prompt(content: str, product_type: str, topics: List[str]) -> str:
    topics_str = "、".join(topics) if topics else "通用"
    return f"""基于以下银行产品文档，生成 {MANUAL_QUERY_COUNT} 个用户可能会问的口语化问题。

要求：
- 所有问题的答案必须能从文档中找到
- 涵盖不同提问角度：定义、条件、流程、限制、对比
- 语言口语化，模拟真实客户

产品类型：{product_type}
相关主题：{topics_str}
文档内容：
{content}

请输出 {MANUAL_QUERY_COUNT} 个问题，每行一个，用序号开头：
1. 
2. 
..."""


def build_process_query_prompt(content: str) -> str:
    return f"""基于以下银行流程中的知识点，生成 {PROCESS_QUERY_COUNT} 个用户可能会问的口语化问题。

要求：
- 问题必须能从给定内容中找到明确答案
- 使用真实客户会用的口语，可以有省略
- 如果内容涉及“客户需要做什么”，就把问题设计成客户在问“我要怎么做”

内容：
{content}

请输出 {PROCESS_QUERY_COUNT} 个问题，每行一个，用序号开头：
1. 
2. 
..."""


def build_regulation_query_prompt(content: str) -> str:
    return f"""基于以下银行政策解读内容，生成 {REGULATION_QUERY_COUNT} 个用户可能会问的口语化问题。

要求：语言口语化，模拟真实客户，问题要能从内容中找到答案。

内容：
{content}

请输出 {REGULATION_QUERY_COUNT} 个问题，每行一个，用序号开头：
1. 
2. 
..."""


def build_glossary_query_prompt(term: str, definition: str, usage: str) -> str:
    return f"""基于以下术语的定义和使用场景，生成 {GLOSSARY_QUERY_COUNT} 个用户口语化问题。

要求：覆盖“是什么”和“对我有什么影响”两种角度。

术语：{term}
定义：{definition}
使用场景：{usage}

请输出 {GLOSSARY_QUERY_COUNT} 个问题，每行一个，用序号开头：
1. 
2. 
..."""


def build_kk_rewrite_prompt(content: str) -> str:
    return f"""将以下银行文档内容，分别改写为 {KK_REWRITE_COUNT} 种不同风格。保持语义不变。

原文：
{content}

1. 客户经理话术（对客户口头解释）：
2. 风险提示短信（简短、正式）：
3. 合同条款说明（严谨、书面）：
（请用数字序号分隔）"""


# ============================================================================
# 负例挖掘器
# ============================================================================

class NegativeMiner:
    """多 Teacher 模型负例挖掘器（优化版）"""

    def __init__(self, teacher_models: Dict[str, Any], chunks: List[Dict]):
        self.models = {}
        self.indexes = {}
        self.doc_embeddings = {}  # 保存每个模型的文档嵌入，避免重复编码
        self.doc_texts = [get_content(c) for c in chunks]
        self.chunks = chunks
        print("加载 Teacher 模型并构建索引...")
        for model_name in teacher_models:
            print(f"  加载 {model_name}...")
            model = SentenceTransformer(model_name, local_files_only=True)
            self.models[model_name] = model
            emb = encode_texts(model, self.doc_texts)
            self.doc_embeddings[model_name] = emb
            self.indexes[model_name] = build_faiss_index(emb)
            print(f"    {model_name} 索引构建完毕，维度={emb.shape[1]}")
        print("所有 Teacher 模型准备就绪。")

    def mine_hard_negatives_batch(
            self,
            queries: List[str],
            positive_chunk_idx: int,
            positive_chunk: Dict,
            use_metadata: bool = True
    ) -> List[Tuple[List[str], List[str]]]:
        """
        批量挖掘多个 query 的困难负例。
        返回：[(hard_negs, easy_negs), ...] 长度与 queries 相同
        """
        positive_text = get_content(positive_chunk)
        pos_product = get_product_type(positive_chunk)
        pos_topics = set(get_topics(positive_chunk))

        # 取第一个模型的嵌入作为过滤用（余弦相似度）
        first_model_name = list(self.models.keys())[0]
        pos_emb = self.doc_embeddings[first_model_name][positive_chunk_idx]

        # 多模型检索，一次性编码所有 query
        all_scores = {}  # model_name -> (scores_matrix, indices_matrix)
        for name, model in self.models.items():
            q_embs = encode_texts(model, queries)  # 批量编码 queries
            scores, indices = self.indexes[name].search(q_embs.astype(np.float32), RETRIEVAL_TOP_K)
            all_scores[name] = (scores, indices)

        results = []
        for q_idx in range(len(queries)):
            # 合并多模型得分
            candidate_scores = defaultdict(float)
            for name, (scores, indices) in all_scores.items():
                for i, idx in enumerate(indices[q_idx]):
                    candidate_scores[int(idx)] += scores[q_idx][i]

            # 剔除正例自身
            if positive_chunk_idx in candidate_scores:
                del candidate_scores[positive_chunk_idx]

            # 按得分排序
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

            # 过滤重复和假负例
            hard_neg_candidates = []
            easy_neg_candidates = []
            for idx, score in sorted_candidates:
                candidate_chunk = self.chunks[idx]
                candidate_text = get_content(candidate_chunk)

                # Jaccard 过滤重复
                if jaccard_similarity(positive_text, candidate_text) > JACCARD_THRESHOLD:
                    continue

                # 余弦相似度过滤假负例（直接使用预存嵌入）
                cand_emb = self.doc_embeddings[first_model_name][idx]
                if cosine_similarity(pos_emb, cand_emb) > COSINE_THRESHOLD:
                    continue

                if use_metadata:
                    cand_product = get_product_type(candidate_chunk)
                    cand_topics = set(get_topics(candidate_chunk))
                    is_hard = False
                    if cand_product == pos_product and pos_topics and cand_topics:
                        if pos_topics != cand_topics and pos_topics & cand_topics:
                            is_hard = True
                    if cand_product != pos_product and pos_topics and cand_topics:
                        if pos_topics == cand_topics:
                            is_hard = True

                    if is_hard:
                        hard_neg_candidates.append(candidate_text)
                    else:
                        easy_neg_candidates.append(candidate_text)
                else:
                    if len(hard_neg_candidates) < HARD_NEG_CANDIDATE_SIZE:
                        hard_neg_candidates.append(candidate_text)
                    else:
                        easy_neg_candidates.append(candidate_text)

            # 困难负例不足时退化为高分文档
            if use_metadata and len(hard_neg_candidates) < QK_HARD_NEG_COUNT:
                for idx, score in sorted_candidates:
                    candidate_text = get_content(self.chunks[idx])
                    if candidate_text not in hard_neg_candidates and jaccard_similarity(positive_text,
                                                                                        candidate_text) <= JACCARD_THRESHOLD:
                        hard_neg_candidates.append(candidate_text)
                        if len(hard_neg_candidates) >= QK_HARD_NEG_COUNT:
                            break

            hard_negs = hard_neg_candidates[:QK_HARD_NEG_COUNT]
            easy_neg_pool = easy_neg_candidates if easy_neg_candidates else [get_content(self.chunks[idx]) for idx, _ in
                                                                             sorted_candidates[-20:]]
            easy_negs = random.sample(easy_neg_pool,
                                      min(QK_EASY_NEG_COUNT, len(easy_neg_pool))) if easy_neg_pool else []
            results.append((hard_negs, easy_negs))

        return results

    def multi_teacher_retrieval(self, query: str, top_k: int = 10):
        """用所有 Teacher 检索，返回 {model_name: (scores, indices)}"""
        results = {}
        for name, model in self.models.items():
            q_emb = encode_texts(model, [query])
            scores, indices = self.indexes[name].search(q_emb.astype(np.float32), top_k)
            results[name] = (scores[0], indices[0])
        return results


# ============================================================================
# 数据生成器
# ============================================================================

class TrainingDataGenerator:
    def __init__(self, chunks: List[Dict], llm_client: LLMClient, negative_miner: NegativeMiner):
        self.chunks = chunks
        self.llm = llm_client
        self.miner = negative_miner
        self.chunk_idx_map = {get_chunk_id(c): i for i, c in enumerate(chunks)}
        self.source_counts = defaultdict(int)
        for c in chunks:
            self.source_counts[get_source_type(c)] += 1
        print(f"数据来源分布: {dict(self.source_counts)}")

    def _parse_numbered_output(self, text: str, count: int) -> List[str]:
        lines = text.strip().split('\n')
        results = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                for sep in ['. ', '.', '、 ', '、', ') ']:
                    if sep in line[:4]:
                        line = line.split(sep, 1)[1].strip()
                        break
                if line:
                    results.append(line)
        return results[:count]

    def generate_qk_data(self) -> List[Dict]:
        print("\n========== 开始生成 Query-Knowledge 数据 ==========")
        all_data = []
        for i, chunk in enumerate(tqdm(self.chunks, desc="QK 生成")):
            source = get_source_type(chunk)
            content = get_content(chunk)
            pos_text = content
            queries = []

            # 构建 queries 列表（与原来完全相同）
            if source == "faq":
                question = get_question(chunk)
                answer = get_answer(chunk)
                if question:
                    queries.append(question)
                    prompt = build_faq_paraphrase_prompt(question, answer if answer else content)
                    resp = self.llm.generate([{"role": "user", "content": prompt}])
                    paraphrases = self._parse_numbered_output(resp, FAQ_PARAPHRASE_COUNT)
                    queries.extend(paraphrases)
            elif source == "product_manual":
                prompt = build_manual_query_prompt(content, get_product_type(chunk), get_topics(chunk))
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries = self._parse_numbered_output(resp, MANUAL_QUERY_COUNT)
            elif source == "process_guide":
                prompt = build_process_query_prompt(content)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries = self._parse_numbered_output(resp, PROCESS_QUERY_COUNT)
            elif source == "regulation":
                prompt = build_regulation_query_prompt(content)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries = self._parse_numbered_output(resp, REGULATION_QUERY_COUNT)
            elif source == "glossary":
                term = get_metadata(chunk).get("term", "")
                definition = get_metadata(chunk).get("definition", content)
                usage = get_metadata(chunk).get("usage", "")
                if term:
                    queries.append(f"什么是{term}？")
                prompt = build_glossary_query_prompt(term, definition, usage)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries.extend(self._parse_numbered_output(resp, GLOSSARY_QUERY_COUNT))
            else:
                continue

            # 跳过无 query 的 chunk
            if not queries:
                continue

            # 批量挖掘负例（核心优化：一次性处理整个 chunk 的所有 queries）
            batch_results = self.miner.mine_hard_negatives_batch(
                queries, i, chunk, use_metadata=True
            )

            # 组装结果
            for q, (hard_negs, easy_negs) in zip(queries, batch_results):
                hard_negs = hard_negs[:QK_HARD_NEG_COUNT]
                easy_negs = easy_negs[:QK_EASY_NEG_COUNT]
                negatives = self._filter_similar_negatives(positive_text=pos_text,negatives=hard_negs + easy_negs)
                if negatives:
                    all_data.append({
                        "query": q,
                        "positive": pos_text,
                        "negatives": negatives
                    })

        print(f"QK 数据生成完毕，共 {len(all_data)} 条")
        return all_data

    def generate_kk_data(self) -> List[Dict]:
        """生成 Knowledge-Knowledge 数据（优化正例构造）"""
        print("\n========== 开始生成 Knowledge-Knowledge 数据 ==========")
        all_data = []

        for i, chunk in enumerate(tqdm(self.chunks, desc="KK 生成")):
            anchor_text = get_content(chunk)
            pt = get_product_type(chunk)
            ts = get_topics(chunk)

            # ----- 正例来源1: Teacher检索 + 元数据过滤 -----
            candidate_indices = self._get_kk_positive_candidates(chunk, i,10)

            # 用 product_type + topics 过滤
            filtered_indices = []
            for idx in candidate_indices:
                cand_chunk = self.chunks[idx]
                if get_product_type(cand_chunk) == pt and set(get_topics(cand_chunk)) == set(ts):
                    filtered_indices.append(idx)

            # 取第一个过滤后的候选；若无则退化为检索的第1个
            if filtered_indices:
                pos_idx = filtered_indices[0]
            elif candidate_indices:
                pos_idx = candidate_indices[0]
            else:
                pos_idx = None  # 极其罕见：Teacher检索完全失败

            # ----- 正例来源2: LLM多角度重写（仅 FAQ 和 Product Manual） -----
            rewrites = []
            source = get_source_type(chunk)
            if source in ("faq", "product_manual"):
                prompt = build_kk_rewrite_prompt(anchor_text)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                parts = resp.split("\n")
                current_style = ""
                for part in parts:
                    part = part.strip()
                    if part.startswith("1.") or part.startswith("2.") or part.startswith("3."):
                        if "：" in part:
                            current_style = part.split("：", 1)[1].strip()
                        elif ": " in part:
                            current_style = part.split(": ", 1)[1].strip()
                        if current_style:
                            rewrites.append(current_style)
                if len(rewrites) < KK_REWRITE_COUNT:
                    rewrites = [anchor_text]  # 降级

            # 构建 positive 文本列表
            positive_texts = []
            if pos_idx is not None:
                positive_texts.append(get_content(self.chunks[pos_idx]))
            positive_texts.extend(rewrites)

            if not positive_texts:
                continue

            # 为每个正例批量挖掘负例
            query_list = [anchor_text] * len(positive_texts)
            batch_results = self.miner.mine_hard_negatives_batch(
                query_list, i, chunk, use_metadata=True
            )

            for pos_text, (hard_negs, easy_negs) in zip(positive_texts, batch_results):
                hard_negs = hard_negs[:KK_HARD_NEG_COUNT]
                easy_negs = easy_negs[:KK_EASY_NEG_COUNT]
                negatives = self._filter_similar_negatives(pos_text, hard_negs + easy_negs)
                if not negatives:
                    continue
                all_data.append({
                    "anchor": anchor_text,
                    "positive": pos_text,
                    "negatives": negatives
                })

        print(f"KK 数据生成完毕，共 {len(all_data)} 条")
        return all_data

    def generate_eval_data(self, qk_count: int = 50, kk_count: int = 50) -> List[Dict]:
        """
        生成评估数据集：QK 和 KK 各指定数量，仅包含正例，不挖掘负例。
        """
        eval_data = []

        # -------------------- 生成 QK 评估数据（保持不变） --------------------
        # qk_candidates = random.sample(self.chunks, min(qk_count, len(self.chunks)))
        # for chunk in tqdm(qk_candidates, desc="生成 QK 评估数据"):
        #     source = get_source_type(chunk)
        #     content = get_content(chunk)
        #     query = None
        #
        #     if source == "faq":
        #         question = get_question(chunk)
        #         if question:
        #             query = question
        #     elif source == "product_manual":
        #         prompt = build_manual_query_prompt(content, get_product_type(chunk), get_topics(chunk))
        #         resp = self.llm.generate([{"role": "user", "content": prompt}])
        #         queries = self._parse_numbered_output(resp, 1)
        #         if queries:
        #             query = queries[0]
        #     elif source == "process_guide":
        #         prompt = build_process_query_prompt(content)
        #         resp = self.llm.generate([{"role": "user", "content": prompt}])
        #         queries = self._parse_numbered_output(resp, 1)
        #         if queries:
        #             query = queries[0]
        #     elif source == "regulation":
        #         prompt = build_regulation_query_prompt(content)
        #         resp = self.llm.generate([{"role": "user", "content": prompt}])
        #         queries = self._parse_numbered_output(resp, 1)
        #         if queries:
        #             query = queries[0]
        #     elif source == "glossary":
        #         term = get_metadata(chunk).get("term", "")
        #         if term:
        #             query = f"什么是{term}？"
        #         else:
        #             prompt = build_glossary_query_prompt(term, content, "")
        #             resp = self.llm.generate([{"role": "user", "content": prompt}])
        #             queries = self._parse_numbered_output(resp, 1)
        #             if queries:
        #                 query = queries[0]
        #
        #     if query:
        #         eval_data.append({"query": query, "positive": content})

        # -------------------- 生成 KK 评估数据（新逻辑） --------------------
        kk_candidates = random.sample(self.chunks, min(kk_count, len(self.chunks)))

        for anchor_chunk in tqdm(kk_candidates, desc="生成 KK 评估数据"):
            anchor_text = get_content(anchor_chunk)
            pt = get_product_type(anchor_chunk)
            ts = get_topics(anchor_chunk)
            anchor_idx = self.chunks.index(anchor_chunk)

            # Teacher 检索候选
            candidate_indices = self._get_kk_positive_candidates(anchor_chunk, anchor_idx,10)

            # 元数据过滤
            filtered_indices = []
            for idx in candidate_indices:
                cand_chunk = self.chunks[idx]
                if get_product_type(cand_chunk) == pt and set(get_topics(cand_chunk)) == set(ts):
                    filtered_indices.append(idx)

            if filtered_indices:
                pos_idx = filtered_indices[0]
            elif candidate_indices:
                pos_idx = candidate_indices[0]
            else:
                continue  # 无候选，跳过

            positive_text = get_content(self.chunks[pos_idx])
            eval_data.append({"anchor": anchor_text, "positive": positive_text})

        print(f"评估数据生成完毕：共 {len(eval_data)} 条 (期望 QK={qk_count}, KK={kk_count})")
        return eval_data

    def _get_kk_positive_candidates(self, anchor_chunk, anchor_idx,top_K):
        """
        使用 Teacher 模型检索语义相近的文档作为正例候选。
        返回候选 chunk 的索引列表（已排除自身且去重）。
        """
        # 收集多 Teacher 模型的检索结果
        all_results = self.miner.multi_teacher_retrieval(get_content(anchor_chunk),top_K)

        candidate_scores = defaultdict(float)
        for model_name, (scores, indices) in all_results.items():
            for idx, score in zip(indices, scores):
                if score > 0.65:
                    candidate_scores[int(idx)] += score

        # 排除自身
        if anchor_idx in candidate_scores:
            del candidate_scores[anchor_idx]

        # 按得分降序排列
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_candidates]

    def _filter_similar_negatives(self,positive_text: str, negatives: List[str], threshold: float = JACCARD_THRESHOLD) -> List[str]:
        return [neg for neg in negatives if jaccard_similarity(positive_text, neg) <= threshold]

    def run(self, mode: str):
        qk_data, kk_data = [], []
        if mode == "other":
            eval_data = self.generate_eval_data(qk_count=70, kk_count=70)
            self._save_data(eval_data, OUTPUT_EVAL)

            kk_data = self.generate_kk_data()
            self._save_data(kk_data, OUTPUT_KK)
        if mode == "eval":
            eval_data = self.generate_eval_data(qk_count=70, kk_count=70)
            self._save_data(eval_data, OUTPUT_EVAL)
        if mode in ("qk", "all"):
            qk_data = self.generate_qk_data()
            self._save_data(qk_data, OUTPUT_QK)
        if mode in ("kk", "all"):
            kk_data = self.generate_kk_data()
            self._save_data(kk_data, OUTPUT_KK)
        if mode == "all":
            all_data = qk_data + kk_data
            self._save_data(all_data, OUTPUT_ALL)
            print(f"混合训练数据已保存至 {OUTPUT_ALL}，总计 {len(all_data)} 条")

    @staticmethod
    def _save_data(data: List[Dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"数据已保存至 {path}，共 {len(data)} 条")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="银行贷款 Embedding 训练数据生成")
    parser.add_argument("--mode", default="all", choices=["qk", "kk", "all","eval","other"],
                        help="生成模式：qk(Query-Knowledge), kk(Knowledge-Knowledge), all(全部)")
    parser.add_argument("--input", default=CHUNK_FILE, help="输入 chunk 文件路径")
    args = parser.parse_args()

    chunks = load_chunks(args.input)
    if not chunks:
        print("没有加载到任何数据，请检查输入文件。")
        return

    llm = LLMClient(api_key=LLM_API_KEY)
    miner = NegativeMiner(TEACHER_MODELS, chunks)
    generator = TrainingDataGenerator(chunks, llm, miner)
    generator.run(args.mode)


if __name__ == "__main__":
    main()
