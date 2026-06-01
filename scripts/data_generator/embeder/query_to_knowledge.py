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
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict

import httpx
import numpy as np
import faiss
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 尝试导入 openai，用于调用 DeepSeek API
try:
    import openai
except ImportError:
    openai = None

# ============================================================================
# 配置区 - 可根据实际情况修改
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 文件路径
CHUNK_FILE = 'chunked_docs.jsonl'
OUTPUT_QK = 'qk_train.jsonl'
OUTPUT_KK = 'kk_train.jsonl'
OUTPUT_vk = 'vk_train.jsonl'
OUTPUT_ALL = 'train_data.jsonl'

model1 = SentenceTransformer("D:/model/models--BAAI--bge-m3", local_files_only=True)
model2 = SentenceTransformer("D:/model/models--GanymedeNil--text2vec-large-chinese", local_files_only=True)
model3 = SentenceTransformer("D:/model/models--moka-ai--m3e-large", local_files_only=True)
# Teacher 模型名称（用于负例挖掘）
TEACHER_MODELS = {
    "BAAI/bge-m3": model1,
    "GanymedeNil/text2vec-large-chinese": model2,
    "moka-ai/m3e-large": model3
}

# 检索 Top-K 设置
RETRIEVAL_TOP_K = 200  # 每个 teacher 检索的候选数
HARD_NEG_CANDIDATE_SIZE = 20  # 从多模型共识中考虑的候选数

# 相似度阈值（用于过滤假负例）
JACCARD_THRESHOLD = 0.8  # Jaccard 相似度超过此值视为重复
COSINE_THRESHOLD = 0.95  # 余弦相似度超过此值视为假负例

# 负例数量配置
QK_HARD_NEG_COUNT = 3
QK_EASY_NEG_COUNT = 2
KK_HARD_NEG_COUNT = 2
KK_EASY_NEG_COUNT = 1

# LLM 配置 (DeepSeek)
LLM_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-v4-flash"

# 数据增强配置
FAQ_PARAPHRASE_COUNT = 2  # FAQ 改写数量
MANUAL_QUERY_COUNT = 2  # Product Manual 生成 query 数量
PROCESS_QUERY_COUNT = 2  # Process Guide 生成 query 数量
REGULATION_QUERY_COUNT = 2  # Regulation 生成 query 数量
GLOSSARY_QUERY_COUNT = 2  # Glossary 生成 query 数量
EVAL_COUNT = 50

# Knowledge-Knowledge 重写数量
KK_REWRITE_COUNT = 3  # LLM 多角度重写版本数
KK_CROSS_PAIR_COUNT = 1  # 跨文档配对数量

# 随机种子，保证可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 批处理大小（用于 Embedding 编码）
BATCH_SIZE = 64


# ============================================================================
# 工具函数
# ============================================================================

def load_chunks(file_path: str) -> List[Dict]:
    """从 jsonl 文件加载所有 chunk"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"从 {file_path} 加载了 {len(chunks)} 个 chunk")
    return chunks


def get_content(chunk: Dict) -> str:
    """获取 chunk 的文本内容"""
    return chunk.get("content", "")


def get_metadata(chunk: Dict) -> Dict:
    """获取 chunk 的 metadata"""
    return chunk.get("metadata", {})


def get_chunk_id(chunk: Dict) -> str:
    """获取 chunk 的唯一标识"""
    meta = get_metadata(chunk)
    return meta.get("chunk_id", hashlib.md5(get_content(chunk).encode()).hexdigest())


def get_product_type(chunk: Dict) -> str:
    """获取 product_type"""
    meta = get_metadata(chunk)
    return meta.get("product_type", "通用")


def get_topics(chunk: Dict) -> List[str]:
    """获取 topics 列表"""
    meta = get_metadata(chunk)
    return meta.get("topics", [])


def get_source_type(chunk: Dict) -> str:
    """获取 source_type"""
    meta = get_metadata(chunk)
    return meta.get("source_type", "")


def get_question(chunk: Dict) -> str:
    """获取 FAQ 的 question 字段"""
    meta = get_metadata(chunk)
    return meta.get("question", "")


def get_answer(chunk: Dict) -> str:
    """获取 FAQ 的 answer 字段"""
    meta = get_metadata(chunk)
    return meta.get("answer", "")


def jaccard_similarity(text1: str, text2: str) -> float:
    """计算两个文本的 Jaccard 相似度"""
    set1 = set(text1)
    set2 = set(text2)
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """批量编码文本，返回 numpy 数组"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """构建内积索引（余弦相似度）"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


# ============================================================================
# LLM 调用封装
# ============================================================================

class LLMClient:
    """DeepSeek API 客户端"""

    def __init__(self, api_key: str, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        if openai is None:
            raise ImportError("请安装 openai 库: pip install openai")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """调用 LLM 生成文本"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body = {"thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return ""


# ============================================================================
# Prompt 模板（根据最终方案定制）
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
# 负例挖掘核心类
# ============================================================================

class NegativeMiner:
    """多 Teacher 模型负例挖掘器"""

    def __init__(self, teacher_models: Dict[str, Any], chunks: List[Dict]):
        self.models = {}
        self.indexes = {}
        self.doc_texts = [get_content(c) for c in chunks]
        self.chunks = chunks
        print("加载 Teacher 模型并构建索引...")
        for model_name, model in teacher_models.items():
            print(f"  加载 {model_name}...")
            self.models[model_name] = model
            embeddings = encode_texts(model, self.doc_texts)
            self.indexes[model_name] = build_faiss_index(embeddings)
            print(f"    {model_name} 索引构建完毕，维度={embeddings.shape[1]}")
        print("所有 Teacher 模型准备就绪。")

    def multi_teacher_retrieval(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> Dict[
        str, Tuple[np.ndarray, np.ndarray]]:
        """用所有 teacher 检索，返回 {model_name: (scores, indices)}"""
        results = {}
        for name, model in self.models.items():
            q_emb = encode_texts(model, [query])
            scores, indices = self.indexes[name].search(q_emb.astype(np.float32), top_k)
            results[name] = (scores[0], indices[0])
        return results

    def mine_hard_negatives(
            self,
            query: str,
            positive_chunk_idx: int,
            positive_chunk: Dict,
            use_metadata: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        挖掘困难负例和简单负例
        返回 (hard_neg_texts, easy_neg_texts)
        """
        positive_text = get_content(positive_chunk)
        pos_product = get_product_type(positive_chunk)
        pos_topics = set(get_topics(positive_chunk))

        # 1. 多 teacher 检索
        all_results = self.multi_teacher_retrieval(query)

        # 2. 收集所有候选索引及其得分
        candidate_scores = defaultdict(float)
        for model_name, (scores, indices) in all_results.items():
            for i, idx in enumerate(indices):
                candidate_scores[int(idx)] += scores[i]

        # 3. 剔除正例自身
        if positive_chunk_idx in candidate_scores:
            del candidate_scores[positive_chunk_idx]

        # 4. 按总得分排序
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        # 5. 过滤重复和假负例
        hard_neg_candidates = []
        easy_neg_candidates = []
        for idx, score in sorted_candidates:
            candidate_chunk = self.chunks[idx]
            candidate_text = get_content(candidate_chunk)

            # Jaccard 过滤重复
            if jaccard_similarity(positive_text, candidate_text) > JACCARD_THRESHOLD:
                continue

            # 余弦相似度过滤假负例（使用任一模型的 embedding 检查）
            pos_emb = encode_texts(list(self.models.values())[0], [positive_text])[0]
            cand_emb = encode_texts(list(self.models.values())[0], [candidate_text])[0]
            if cosine_similarity(pos_emb, cand_emb) > COSINE_THRESHOLD:
                continue

            # Metadata 筛选（如果启用）
            if use_metadata:
                cand_product = get_product_type(candidate_chunk)
                cand_topics = set(get_topics(candidate_chunk))

                # 困难负例规则：
                # 规则1: 同一 product_type + 相邻 topics（有交集但不完全重叠）
                # 规则2: 同一 topics + 不同 product_type
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
                    # 不符合规则的作为简单负例候选
                    easy_neg_candidates.append(candidate_text)
            else:
                # 无 metadata 退化：前几条为困难，后面的为简单
                if len(hard_neg_candidates) < HARD_NEG_CANDIDATE_SIZE:
                    hard_neg_candidates.append(candidate_text)
                else:
                    easy_neg_candidates.append(candidate_text)

        # 如果启用 metadata 但没找到足够困难负例，退化为取高分前几条
        if use_metadata and len(hard_neg_candidates) < QK_HARD_NEG_COUNT:
            # 退化为纯得分前几条（已过滤重复）
            for idx, score in sorted_candidates:
                candidate_text = get_content(self.chunks[idx])
                if candidate_text not in hard_neg_candidates and jaccard_similarity(positive_text,
                                                                                    candidate_text) <= JACCARD_THRESHOLD:
                    hard_neg_candidates.append(candidate_text)
                    if len(hard_neg_candidates) >= QK_HARD_NEG_COUNT:
                        break

        # 选取指定数量
        hard_negs = hard_neg_candidates[:QK_HARD_NEG_COUNT]  # 默认用 QK 配置，KK 可调用后截断
        # 简单负例从末尾随机选
        easy_neg_pool = easy_neg_candidates if easy_neg_candidates else [get_content(self.chunks[idx]) for idx, _ in
                                                                         sorted_candidates[-20:]]
        easy_negs = random.sample(easy_neg_pool, min(QK_EASY_NEG_COUNT, len(easy_neg_pool))) if easy_neg_pool else []

        return hard_negs, easy_negs


# ============================================================================
# 数据生成主类
# ============================================================================

class TrainingDataGenerator:
    def __init__(self, chunks: List[Dict], llm_client: LLMClient, negative_miner: NegativeMiner):
        self.chunks = chunks
        self.llm = llm_client
        self.miner = negative_miner
        self.chunk_idx_map = {get_chunk_id(c): i for i, c in enumerate(chunks)}

        # 统计各来源数量
        self.source_counts = defaultdict(int)
        for c in chunks:
            self.source_counts[get_source_type(c)] += 1
        print(f"数据来源分布: {dict(self.source_counts)}")

    def _parse_numbered_output(self, text: str, count: int) -> List[str]:
        """解析 LLM 返回的序号列表"""
        lines = text.strip().split('\n')
        results = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # 移除序号前缀 (如 "1. " 或 "1、")
                for separator in ['. ', '.', '、 ', '、', ') ']:
                    if separator in line[:4]:
                        line = line.split(separator, 1)[1].strip()
                        break
                if line:
                    results.append(line)
        return results[:count]

    def generate_qk_data(self) -> List[Dict]:
        """生成 Query-Knowledge 数据"""
        print("\n========== 开始生成 Query-Knowledge 数据 ==========")
        all_data = []

        for i, chunk in enumerate(tqdm(self.chunks, desc="QK 生成")):
            source = get_source_type(chunk)
            content = get_content(chunk)
            pos_text = content  # 正例文档即为 content

            queries = []
            if source == "faq":
                question = get_question(chunk)
                answer = get_answer(chunk)
                if question:
                    queries.append(question)
                    # Paraphrase 增强
                    prompt = build_faq_paraphrase_prompt(question, answer if answer else content)
                    resp = self.llm.generate([{"role": "user", "content": prompt}])
                    paraphrases = self._parse_numbered_output(resp, FAQ_PARAPHRASE_COUNT)
                    queries.extend(paraphrases)
            elif source == "product_manual":
                prompt = build_manual_query_prompt(
                    content,
                    get_product_type(chunk),
                    get_topics(chunk)
                )
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
                # 术语本身作为 query
                if term:
                    queries.append(f"什么是{term}？")
                prompt = build_glossary_query_prompt(term, definition, usage)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries.extend(self._parse_numbered_output(resp, GLOSSARY_QUERY_COUNT))
            else:
                # 未知来源，跳过
                continue

            # 为每个 query 挖掘负例
            for q in queries:
                if not q:
                    continue
                hard_negs, easy_negs = self.miner.mine_hard_negatives(
                    q, i, chunk, use_metadata=True
                )
                # 确保数量
                hard_negs = hard_negs[:QK_HARD_NEG_COUNT]
                easy_negs = easy_negs[:QK_EASY_NEG_COUNT]
                negatives = hard_negs + easy_negs
                if not negatives:
                    continue  # 没负例则跳过这条 query

                all_data.append({
                    "query": q,
                    "positive": pos_text,
                    "negatives": negatives
                })

        print(f"QK 数据生成完毕，共 {len(all_data)} 条")
        return all_data

    def generate_kk_data(self) -> List[Dict]:
        """生成 Knowledge-Knowledge 数据"""
        print("\n========== 开始生成 Knowledge-Knowledge 数据 ==========")
        all_data = []

        # 构建 product_type 和 topics 的索引，用于快速查找同主题文档
        topic_index = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            pt = get_product_type(chunk)
            ts = get_topics(chunk)
            key = (pt, tuple(sorted(ts)))
            topic_index[key].append(i)

        for i, chunk in enumerate(tqdm(self.chunks, desc="KK 生成")):
            anchor_text = get_content(chunk)
            pt = get_product_type(chunk)
            ts = get_topics(chunk)

            # 正例来源1: 同一 product_type + 同一 topics 的另一文档
            same_topic_indices = []
            for key, indices in topic_index.items():
                if key[0] == pt and set(key[1]) == set(ts):
                    same_topic_indices.extend(indices)
            # 排除自身
            same_topic_indices = [j for j in same_topic_indices if j != i]

            # 正例来源2: LLM 多角度重写（仅 FAQ 和 Product Manual 做）
            rewrites = []
            source = get_source_type(chunk)
            if source in ("faq", "product_manual"):
                prompt = build_kk_rewrite_prompt(anchor_text)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                # 解析出3个版本
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
                    # 如果解析失败，手动构造几条
                    rewrites = [anchor_text]  # 降级

            # 构建多个正例对
            positive_texts = []
            # 同主题文档正例（取1个）
            if same_topic_indices:
                pos_idx = random.choice(same_topic_indices)
                positive_texts.append(get_content(self.chunks[pos_idx]))
            # 重写版本正例
            for rw in rewrites:
                positive_texts.append(rw)

            # 如果没有正例，跳过
            if not positive_texts:
                continue

            # 为每个正例挖掘负例
            for pos_text in positive_texts:
                # 使用 anchor 作为 query 去挖掘负例
                hard_negs, easy_negs = self.miner.mine_hard_negatives(
                    anchor_text, i, chunk, use_metadata=True
                )
                hard_negs = hard_negs[:KK_HARD_NEG_COUNT]
                easy_negs = easy_negs[:KK_EASY_NEG_COUNT]
                negatives = hard_negs + easy_negs
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
        返回格式：
        [
            {"query": "用户问题", "positive": "文档片段"},        # QK 评估样本
            {"anchor": "文档片段A", "positive": "文档片段B"},     # KK 评估样本
        ]
        """
        eval_data = []

        # -------------------- 生成 QK 评估数据 --------------------
        # 从所有 chunk 中随机采样，确保每个样本独立（一个 chunk 只生成一条 query）
        qk_candidates = random.sample(self.chunks, min(qk_count, len(self.chunks)))

        for chunk in tqdm(qk_candidates, desc="生成 QK 评估数据"):
            source = get_source_type(chunk)
            content = get_content(chunk)
            query = None

            if source == "faq":
                question = get_question(chunk)
                if question:
                    query = question
            elif source == "product_manual":
                prompt = build_manual_query_prompt(content, get_product_type(chunk), get_topics(chunk))
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries = self._parse_numbered_output(resp, 1)  # 只要1条
                if queries:
                    query = queries[0]
            elif source == "process_guide":
                prompt = build_process_query_prompt(content)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries = self._parse_numbered_output(resp, 1)
                if queries:
                    query = queries[0]
            elif source == "regulation":
                prompt = build_regulation_query_prompt(content)
                resp = self.llm.generate([{"role": "user", "content": prompt}])
                queries = self._parse_numbered_output(resp, 1)
                if queries:
                    query = queries[0]
            elif source == "glossary":
                term = get_metadata(chunk).get("term", "")
                if term:
                    query = f"什么是{term}？"
                else:
                    prompt = build_glossary_query_prompt(term, content, "")
                    resp = self.llm.generate([{"role": "user", "content": prompt}])
                    queries = self._parse_numbered_output(resp, 1)
                    if queries:
                        query = queries[0]

            if query:
                eval_data.append({"query": query, "positive": content})

        # -------------------- 生成 KK 评估数据 --------------------
        # 构建 topic 索引（同 generate_kk_data）
        topic_index = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            pt = get_product_type(chunk)
            ts = get_topics(chunk)
            topic_index[(pt, tuple(sorted(ts)))].append(i)

        # 随机选 kk_count 个 anchor chunk
        kk_candidates = random.sample(self.chunks, min(kk_count, len(self.chunks)))

        for anchor_chunk in tqdm(kk_candidates, desc="生成 KK 评估数据"):
            anchor_text = get_content(anchor_chunk)
            pt = get_product_type(anchor_chunk)
            ts = get_topics(anchor_chunk)

            # 找到同主题的候选文档（排除自身）
            same_topic_indices = []
            for key, indices in topic_index.items():
                if key[0] == pt and set(key[1]) == set(ts):
                    same_topic_indices.extend(indices)
            same_topic_indices = [j for j in same_topic_indices if j != self.chunks.index(anchor_chunk)]

            # 降级：如果没有同主题，使用 Teacher 语义检索一个相近文档
            if not same_topic_indices:
                all_results = self.miner.multi_teacher_retrieval(anchor_text)
                candidate_scores = defaultdict(float)
                for model_name, (scores, indices) in all_results.items():
                    for idx, score in zip(indices, scores):
                        candidate_scores[int(idx)] += score
                anchor_idx = self.chunks.index(anchor_chunk)
                if anchor_idx in candidate_scores:
                    del candidate_scores[anchor_idx]
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                for idx, score in sorted_candidates:
                    cand_text = get_content(self.chunks[idx])
                    if jaccard_similarity(anchor_text, cand_text) <= JACCARD_THRESHOLD:
                        same_topic_indices.append(idx)
                        break

            if same_topic_indices:
                pos_idx = random.choice(same_topic_indices)
                positive_text = get_content(self.chunks[pos_idx])
                eval_data.append({"anchor": anchor_text, "positive": positive_text})
            else:
                # 实在找不到，跳过
                continue

        print(f"评估数据生成完毕：共 {len(eval_data)} 条 (期望 QK={qk_count}, KK={kk_count})")
        return eval_data

    def run(self, mode: str):
        """运行数据生成流水线"""
        qk_data = []
        kk_data = []

        if mode == 'eval':
            val_data = self.generate_eval_data()
            self._save_data(val_data,OUTPUT_vk)

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

    def _save_data(self, data: List[Dict], path: str):
        """保存数据为 jsonl"""
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"数据已保存至 {path}，共 {len(data)} 条")


# ============================================================================
# 主函数
# ============================================================================

def main(args):
    # 加载数据
    chunks = load_chunks(args.input)
    if not chunks:
        print("没有加载到任何数据，请检查输入文件。")
        return

    # 初始化 LLM 客户端
    llm = LLMClient(api_key=LLM_API_KEY)

    # 初始化 Teacher 模型和负例挖掘器
    miner = NegativeMiner(TEACHER_MODELS, chunks)

    # 初始化数据生成器
    generator = TrainingDataGenerator(chunks, llm, miner)

    # 开始生成
    generator.run(args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="银行贷款 Embedding 训练数据生成")
    parser.add_argument("--mode", default="all",help="生成模式：qk(Query-Knowledge), kk(Knowledge-Knowledge), all(全部)")
    parser.add_argument("--input", default=CHUNK_FILE, help="输入 chunk 文件路径")
    args = parser.parse_args()
    main(args)
