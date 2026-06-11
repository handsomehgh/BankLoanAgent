# author hgh
# version 1.0
#!/usr/bin/env python3
"""
general_search_knowledge 训练数据生成脚本

定位：BERT 分类的兜底选项（层级3）
- 不能从上下文中直接回答（非 DIRECT_REPLY）
- 不能匹配到具体的计算/查询/意向工具（非具体工具/Skill）
- 也不是完全无法理解意图（非 CLARIFY）
→ 选择 general_search_knowledge，先去知识库检索，结果交给 LLM 处理。

生成策略：
  来源一（基于 chunk 生成，约60%）：随机抽取 faq/product_manual 的 chunk，
    不将 chunk 内容放入 text_a，让 LLM 基于 chunk 生成需要检索才能回答的问题。
  来源二（自由生成，约40%）：LLM 自由生成合理的贷款业务咨询问题，
    不依赖具体 chunk，但确保不属于计算/查个人数据/提交操作。

关键：text_a 的 business_knowledge 字段永远为空。
输出：general_search_knowledge_samples.jsonl
"""

import json
import random
import time
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

# ======================== 配置 ========================
DEFAULT_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# 总样本数
TOTAL_COUNT = 30
SOURCE1_RATIO = 0.6  # 基于chunk生成的比例
# 各类样本数量
SOURCE1_COUNT = int(TOTAL_COUNT * SOURCE1_RATIO)   # 90
SOURCE2_COUNT = TOTAL_COUNT - SOURCE1_COUNT        # 60

RANDOM_SEED = 42

# ======================== 多轮上下文模板 ========================
PROFILES = [
    "用户画像：月收入约3万元，名下一套房无贷款。",
    "用户画像：月收入1.2万，有消费贷月供2000元。",
    "用户画像：个体经营者，年收入50万，征信有1次逾期已结清。",
    "用户画像：公司职员，月收入2.5万，公积金缴存基数2万。",
    "用户画像：退休工程师，月退休金1万元，有存款80万。",
    "用户画像：28岁，互联网从业者，月薪4万，无房无贷。",
    "用户画像：离异，月收入1.8万，名下无房。",
    "用户画像：小微企业合伙人，年分红30万，有经营贷负债40万。",
    "用户画像：设计师，自由职业，近半年月均收入2.5万。",
    "用户画像：公务员，月收入1.6万，公积金余额20万，首次购房。",
]

SUMMARIES = [
    "对话摘要：用户询问了公积金贷款上限，助手根据缴存基数和余额进行估算。",
    "对话摘要：用户表示征信报告上有一次信用卡逾期，担心影响房贷审批。",
    "对话摘要：用户咨询了装修贷款的最高额度，助手介绍了不同产品方案。",
    "对话摘要：用户询问能否将商业贷款转为公积金贷款，助手解释了商转公的条件。",
    "对话摘要：用户想了解贷款期间能否出售抵押房产，助手说明需先还清贷款。",
    "对话摘要：用户对利率下调后月供是否变化有疑问，助手解释了重定价机制。",
    "对话摘要：用户询问了房贷审批不通过的常见原因。",
    "对话摘要：用户咨询了等额本息改为等额本金的可行性，助手说明需要重新审批。",
    "对话摘要：用户询问了贷款用途凭证的保留要求，助手说明需保留发票和合同。",
    "对话摘要：用户想了解夫妻离婚后贷款责任如何划分。",
]

RECENT_CONVS = [
    "最近对话：用户: 商贷和公积金组合贷款怎么办理？\n助手: 需要分别向银行和公积金中心申请。",
    "最近对话：用户: 我的征信报告上显示有一次逾期，会影响房贷审批吗？\n助手: 要看逾期程度和距今时间。",
    "最近对话：用户: 贷款年限最长能选多少年？\n助手: 住房贷款最长30年，消费贷一般3-5年。",
    "最近对话：用户: 收入证明需要盖什么章？\n助手: 需要单位公章或人事章，并注明月收入金额。",
    "最近对话：用户: 等额本息中途能改等额本金吗？\n助手: 可以申请，但需要重新审批，可能涉及手续费。",
    "最近对话：用户: 贷款审批没通过，多久能再申请？\n助手: 一般建议3-6个月后，先改善征信或降低负债。",
    "最近对话：用户: 我名下有房，再买一套首付要多少？\n助手: 二套房首付一般不低于60%，具体看当地政策。",
    "最近对话：用户: 贷款下来后可以提前还一部分吗？\n助手: 可以，满一年后申请不收违约金。",
    "最近对话：用户: 我换工作了，试用期能贷款吗？\n助手: 一般要求工作满6个月，试用期可能受影响。",
    "最近对话：用户: 经营贷的用途有限制吗？\n助手: 只能用于企业经营周转，不能用于购房或投资。",
]

TOOL_OPS = [
    "工具操作：助手: query_interest_rate\n工具结果: 消费贷年利率4.5%起。",
    "工具操作：助手: calculate_monthly_payment\n工具结果: 贷款80万20年，月供约5100元。",
    "工具操作：助手: check_loan_eligibility\n工具结果: 因近期征信查询次数过多，建议3个月后再申请。",
    "工具操作：助手: calculate_max_loan_amount\n工具结果: 基于月收入2.5万，最高可贷约120万。",
    "工具操作：助手: compare_loan_products\n工具结果: 等额本金比等额本息节省总利息约15%。",
    "工具操作：助手: general_search_knowledge\n工具结果: 经营贷申请条件：营业执照满2年，经营流水。",
    "工具操作：助手: calculate_loan_total_cost\n工具结果: 总成本含评估费3000元、保险费2000元。",
    "工具操作：助手: generate_repayment_schedule\n工具结果: 已生成60期还款计划，每月本金和利息逐月变化。",
    "工具操作：助手: apply_home_loan_skill\n工具结果: 综合评估：参考利率4.2%，建议贷款7成。",
    "工具操作：助手: apply_consumer_loan_skill\n工具结果: 消费贷额度20万，推荐3年期，利率4.8%。",
]

MIXED_RICH = [
    "用户画像：公务员，月收入1.6万，公积金余额20万。\n对话摘要：用户询问了贷款期间能否出售抵押房产。\n最近对话：用户: 如果我把房子卖了，贷款怎么办？\n助手: 需要先还清贷款解除抵押才能过户。",
    "用户画像：个体经营者，年收入50万，征信有1次逾期已结清。\n对话摘要：用户咨询了经营贷的申请条件。\n工具操作：助手: general_search_knowledge\n工具结果: 经营贷申请需提供营业执照满2年。",
    "用户画像：28岁，互联网从业者，月薪4万。\n对话摘要：用户想了解等额本息和等额本金哪个更适合。\n最近对话：用户: 我收入比较稳定，选哪种还款方式好？\n助手: 等额本金总利息少，但前期月供高。",
    "用户画像：退休工程师，月退休金1万元，有存款80万。\n对话摘要：用户咨询了装修贷款的最高额度。\n最近对话：用户: 我想把老房子重新装修一下。\n助手: 装修贷属于消费贷，您可以先了解申请条件和额度。",
    "用户画像：设计师，自由职业，近半年月均收入2.5万。\n对话摘要：用户对贷款审批不通过的原因有疑问。\n工具操作：助手: check_loan_eligibility\n工具结果: 因工作证明不够充分，建议提供近6个月银行流水。",
    "用户画像：小微企业合伙人，年分红30万，有经营贷负债40万。\n对话摘要：用户咨询了利率下调后月供的变化。\n最近对话：用户: LPR降了，我的月供能跟着降吗？\n助手: 浮动利率贷款会在重定价日自动调整。",
    "用户画像：离异，月收入1.8万，名下无房。\n对话摘要：用户询问了离婚后贷款责任的划分。\n最近对话：用户: 房子归前妻，贷款还在我名下，怎么办？\n助手: 需要办理贷款变更手续，由实际居住方承担还款责任。",
    "用户画像：公司职员，月收入2.5万，公积金缴存基数2万。\n对话摘要：用户咨询了公积金冲还贷的办理条件。\n工具操作：助手: general_search_knowledge\n工具结果: 公积金冲还贷需连续缴存满6个月，可在公积金中心或手机APP签约。",
    "用户画像：月收入约3万元，名下一套房无贷款。\n对话摘要：用户想了解贷款期间更换还款账户的流程。\n最近对话：用户: 我换了张工资卡，怎么改还款账户？\n助手: 需要本人携带新卡和身份证到柜台办理变更。",
    "用户画像：月收入1.2万，有消费贷月供2000元。\n对话摘要：用户对贷款用途凭证的保留有疑问。\n最近对话：用户: 消费贷的钱用了之后要保留什么凭证？\n助手: 建议保留发票、合同等，银行可能会抽查用途。",
]
# ======================== 工具函数 ========================
def load_chunks(file_path: str = "chunked_docs.jsonl", filter_source: bool = True) -> List[Dict]:
    """加载 chunked_docs.jsonl，可选过滤 source_type=faq/product_manual"""
    chunks = []
    if not os.path.exists(file_path):
        print(f"警告：{file_path} 不存在")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if filter_source:
                source_type = chunk.get("metadata", {}).get("source_type", "")
                if source_type not in ("faq", "product_manual"):
                    continue
            chunks.append(chunk)
    print(f"加载了 {len(chunks)} 个符合 source_type 的 chunk (faq/product_manual)")
    return chunks

def get_random_chunk(chunks: List[Dict]) -> Optional[Dict]:
    if not chunks:
        return None
    return random.choice(chunks)

def build_base_context(with_profile=True, with_summary=True, with_tools=True, with_recent=True) -> str:
    """构造不含业务知识的上下文（四个字段）"""
    parts = []
    parts.append(f"用户画像：{random.choice(PROFILES) if with_profile else '暂无相关信息'}")
    parts.append(f"对话摘要：{random.choice(SUMMARIES) if with_summary else '暂无相关信息'}")
    parts.append(f"近期工具操作：{random.choice(TOOL_OPS) if with_tools else '暂无相关信息'}")
    parts.append(f"最近对话：{random.choice(RECENT_CONVS) if with_recent else '暂无相关信息'}")
    return "\n".join(parts)

def build_text_a(base_context: str, business_knowledge: str = "暂无相关信息") -> str:
    """拼接完整 text_a（五个字段），business_knowledge 固定为空"""
    return f"{base_context}\n业务知识：{business_knowledge}"

# ======================== LLM 调用 ========================
def query_llm_for_source1(client: OpenAI, model: str, chunk_content: str, base_context: str) -> Optional[str]:
    """来源一：基于 chunk 生成问题，上下文不得包含答案"""
    text_a = build_text_a(base_context, "暂无相关信息")
    prompt = f"""你是一个正在咨询银行贷款的用户。

下面是银行知识库中的一段业务知识（仅供内部参考，不会提供给用户）：
{chunk_content[:800]}

现有对话上下文（其中业务知识为空）：
{text_a}

请生成一个用户提问，要求：
1. 该提问的答案必须能从上述业务知识中找到
2. 该提问不能从已有上下文（用户画像、对话摘要、近期工具操作、最近对话）中直接回答
3. 该提问不能是计算类问题（如算月供、算额度、算总成本）
4. 该提问不能是查询个人数据（如查我的贷款进度、查我的利率、查我的意向）
5. 该提问不能是提交/修改操作（如帮我提交申请、改金额、改期限）
6. 该提问不能是纯情绪表达或确认（如“好的”、“压力好大”）
7. 该提问必须是一个需要银行专业知识才能回答的问题
8. 口语化、自然，只输出问题本身

如果无法生成合适的问题，请只输出：SKIP"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=100,
            )
            result = resp.choices[0].message.content.strip()
            if result == "SKIP":
                return None
            if result and len(result) >= 4:
                return result
        except Exception as e:
            print(f"LLM 调用失败，重试 {attempt+1}: {e}")
            time.sleep(2)
    return None

def query_llm_for_source2(client: OpenAI, model: str, base_context: str) -> Optional[str]:
    """来源二：自由生成合理的贷款业务咨询问题"""
    text_a = build_text_a(base_context, "暂无相关信息")
    prompt = f"""你是一个正在咨询银行贷款的用户。

现有对话上下文（其中业务知识为空）：
{text_a}

请生成一个合理的银行贷款业务咨询问题，要求：
1. 该问题是一个需要银行专业知识或政策才能完整回答的问题
2. 不能是计算类问题（如算月供、算额度、算总成本）
3. 不能是查询个人数据（如查我的贷款进度、查我的利率、查我的意向）
4. 不能是提交/修改操作（如帮我提交申请、改金额、改期限）
5. 不能是纯情绪表达或确认
6. 不能是完全无法理解意图的模糊问题（如“帮我看一下”、“那个呢”）
7. 问题应该清晰、明确，但需要检索知识库才能给出准确答案
8. 口语化、自然，只输出问题本身

如果无法生成合适的问题，请只输出：SKIP"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=100,
            )
            result = resp.choices[0].message.content.strip()
            if result == "SKIP":
                return None
            if result and len(result) >= 4:
                return result
        except Exception as e:
            print(f"LLM 调用失败，重试 {attempt+1}: {e}")
            time.sleep(2)
    return None

# ======================== 生成函数 ========================
def generate_source1(client: OpenAI, model: str, chunks: List[Dict], count: int) -> List[Dict]:
    """来源一：基于 chunk 生成"""
    samples = []
    attempts = 0
    max_attempts = count * 3
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        chunk = get_random_chunk(chunks)
        if not chunk:
            break
        content = chunk.get("content", "")
        if not content:
            continue
        # 随机决定是单轮还是多轮
        is_single = random.random() < 0.3  # 30% 单轮
        if is_single:
            base_context = "用户画像：暂无相关信息\n对话摘要：暂无相关信息\n近期工具操作：暂无相关信息\n最近对话：暂无相关信息"
        else:
            base_context = build_base_context(
                with_profile=random.random() > 0.3,
                with_summary=random.random() > 0.3,
                with_tools=random.random() > 0.5,
                with_recent=random.random() > 0.2
            )
        question = query_llm_for_source1(client, model, content, base_context)
        if question:
            text_a = build_text_a(base_context, "暂无相关信息")
            samples.append({"text_a": text_a, "text_b": question, "label": "general_search_knowledge"})
            print(f"来源一 生成 {len(samples)}/{count}")
    return samples

def generate_source2(client: OpenAI, model: str, count: int) -> List[Dict]:
    """来源二：自由生成"""
    samples = []
    attempts = 0
    max_attempts = count * 3
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        is_single = random.random() < 0.3
        if is_single:
            base_context = "用户画像：暂无相关信息\n对话摘要：暂无相关信息\n近期工具操作：暂无相关信息\n最近对话：暂无相关信息"
        else:
            base_context = build_base_context(
                with_profile=random.random() > 0.3,
                with_summary=random.random() > 0.3,
                with_tools=random.random() > 0.5,
                with_recent=random.random() > 0.2
            )
        question = query_llm_for_source2(client, model, base_context)
        if question:
            text_a = build_text_a(base_context, "暂无相关信息")
            samples.append({"text_a": text_a, "text_b": question, "label": "general_search_knowledge"})
            print(f"来源二 生成 {len(samples)}/{count}")
    return samples

# ======================== 主函数 ========================
def main(args):
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    model = args.model
    random.seed(RANDOM_SEED)

    # 加载 chunk（仅 faq 和 product_manual）
    chunks = load_chunks(args.chunk_file, filter_source=True)
    if not chunks:
        print("未加载到 chunk，退出。")
        return

    all_samples = []

    print(f"来源一：基于 chunk 生成，目标 {SOURCE1_COUNT} 条...")
    source1 = generate_source1(client, model, chunks, SOURCE1_COUNT)
    all_samples.extend(source1)

    print(f"来源二：自由生成，目标 {SOURCE2_COUNT} 条...")
    source2 = generate_source2(client, model, SOURCE2_COUNT)
    all_samples.extend(source2)

    # 去重
    seen = set()
    unique = []
    for item in all_samples:
        key = (item["text_a"], item["text_b"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    all_samples = unique

    # 保存
    output_file = args.output or "general_search_knowledge_samples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n生成完毕，共 {len(all_samples)} 条 general_search_knowledge 样本，保存至 {output_file}")
    print(f"  来源一: {len(source1)}")
    print(f"  来源二: {len(source2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--chunk_file", type=str, default="chunked_docs.jsonl")
    parser.add_argument("--output", type=str, default="advisor_knowledge_samples.jsonl")
    args = parser.parse_args()
    main(args)
