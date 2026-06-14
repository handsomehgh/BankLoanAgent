#!/usr/bin/env python3
"""
DIRECT_REPLY 训练数据生成脚本（均衡严苛版）
生成策略：
  1. 基于业务知识（business_knowledge 有内容）- 约 20%
  2. 基于上下文（business_knowledge 空，答案在画像/摘要/工具操作/最近对话）- 约 60%
  3. 干扰项（business_knowledge 有内容但与问题无关，答案在上下文其他部分或常识）- 约 20%
核心原则：LLM 必须在严格判断下生成问题，答案必须明确可从上下文中直接获取，
避免仅因少量关键词重叠就认定可回答，否则输出 SKIP。
"""

import json
import random
import time
import argparse
import os
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# ======================== 配置 ========================
DEFAULT_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

TOTAL_SAMPLES = 200
RANDOM_SEED = 42

# ======================== 丰富上下文模板（各20条） ========================
PROFILES = [
    "用户画像：月收入约2万元，无负债。",
    "用户画像：月收入1.5万，有车贷3000元/月。",
    "用户画像：个体户，年收入30万，征信良好。",
    "用户画像：自由职业，月均入账2万但无正式流水。",
    "用户画像：退休人员，月退休金8000元。",
    "用户画像：国企员工，月收入1.8万，公积金缴存基数1.5万，有房贷月供4000元。",
    "用户画像：25岁，刚工作1年，月收入8000元，无负债，信用记录空白。",
    "用户画像：已婚，家庭月收入合计4万元，名下一套房贷款已还清，欲购二套房。",
    "用户画像：小微企业主，年营业额200万，月均净利润5万，有经营贷负债30万。",
    "用户画像：自由撰稿人，收入不稳定，近6个月平均月入1.2万，无资产。",
    "用户画像：教师，月收入1万元，公积金缴存比例12%，名下无房，首次购房。",
    "用户画像：快递员，月收入9000元，现金结算无社保，有花呗欠款5000元。",
    "用户画像：35岁，互联网工程师，月薪3.5万，持有股票期权，负债为信用卡2万。",
    "用户画像：宝妈，无固定工作，配偶月入3万，家庭名下无贷款，想以个人名义申请消费贷。",
    "用户画像：退休公务员，月退休金1.2万元，有定期存款50万，征信无逾期。",
    "用户画像：刚毕业博士生，入职3个月，月薪1.6万，无负债，意向申请住房贷款。",
    "用户画像：外卖骑手，月收入不稳定，最近3个月均收入1.1万，无劳动合同。",
    "用户画像：小卖部店主，月均净利1.5万，无营业执照贷款记录，征信有2次逾期已结清。",
    "用户画像：医生，月收入2.5万，公积金基数2.2万，已有房一套，欲购学区房。",
    "用户画像：出租车司机，月入1.3万，有车无负债，但收入难以提供流水。",
]

SUMMARIES = [
    "对话摘要：用户刚咨询了房贷利率，助手回复首套房LPR 4.2% + 30BP。",
    "对话摘要：用户询问消费贷额度，助手表示需要评估收入和负债。",
    "对话摘要：用户想了解经营贷和消费贷的区别，助手对比了利率和期限。",
    "对话摘要：用户表达了想申请装修贷款的意愿，助手介绍了基本条件。",
    "对话摘要：用户担心征信不良会影响贷款，助手询问了具体逾期情况。",
    "对话摘要：用户询问了公积金贷款流程，助手说明了申请条件和所需材料。",
    "对话摘要：用户对比了抵押贷和信用贷的优劣，助手建议根据额度需求选择。",
    "对话摘要：用户询问贷款审批时效，助手回复信用贷1-3天，房贷约1-2周。",
    "对话摘要：用户询问了提前还款政策，助手说明满一年免收违约金。",
    "对话摘要：用户想了解组合贷款（公积金+商贷）的办理方式，助手解释了比例限制。",
    "对话摘要：用户咨询了贷款利率与LPR的关系，助手解释了加点机制。",
    "对话摘要：用户询问了贷款被拒后的再申请策略，助手建议改善征信和降低负债。",
    "对话摘要：用户对还款计划表有疑问，助手解释了等额本息的利息计算方式。",
    "对话摘要：用户询问了贷款用途的限制，助手明确禁止用于购房首付或投资。",
    "对话摘要：用户询问了线上申请贷款的额度限制，助手答复纯信用消费贷最高30万。",
    "对话摘要：用户询问了助学贷款或留学贷款的可能性，助手介绍了教育贷款产品。",
    "对话摘要：用户询问了车位贷或装修贷的具体利率和期限，助手给出了参考区间。",
    "对话摘要：用户询问了贷款保险的必要性，助手解释房贷通常强制购买抵押物保险。",
    "对话摘要：用户询问了贷款期间能否增加共同借款人，助手说明需要重新审批。",
    "对话摘要：用户咨询了房贷转按揭到其他银行的流程和费用。",
]

RECENT_CONVS = [
    "最近对话：用户: 那贷100万30年月供呢？\n助手: 请提供年利率和还款方式。",
    "最近对话：用户: 能贷多少？\n助手: 您月收入和现有负债大概多少？",
    "最近对话：用户: 等额本息和等额本金哪个更省钱？\n助手: 等额本金总利息更少。",
    "最近对话：用户: 利率怎么这么高？\n助手: 目前的利率是根据您的征信和产品类型定的。",
    "最近对话：用户: 材料已经提交了，什么时候能审批下来？\n助手: 通常在3个工作日内。",
    "最近对话：用户: 我想贷20万装修，5年还，月供别超过4000行吗？\n助手: 我们试算一下。",
    "最近对话：用户: 我征信有过一次逾期，影响大吗？\n助手: 单次非连续逾期且已结清，影响有限。",
    "最近对话：用户: 公积金余额能用来付首付吗？\n助手: 一般不能直接用于首付。",
    "最近对话：用户: 我老婆没有工作，我们还能贷款吗？\n助手: 可以用您的收入作为主要还款来源。",
    "最近对话：用户: 还款日我忘了，今天才补上，算逾期吗？\n助手: 一般在还款日后3天内算宽限期。",
    "最近对话：用户: 我想把等额本息改成等额本金可以吗？\n助手: 还款方式变更需要申请。",
    "最近对话：用户: 我在外地有套房，再买一套算首套吗？\n助手: 认房又认贷的城市可能算二套。",
    "最近对话：用户: 贷款批下来钱打到哪里？\n助手: 房贷会直接划入开发商或卖方账户。",
    "最近对话：用户: 我已经有两张信用卡了，再贷款会不会影响审批？\n助手: 信用卡已用额度会算入负债。",
    "最近对话：用户: 我月收入1万多，但都是现金，没有流水，能贷款吗？\n助手: 可以提供其他资产证明。",
    "最近对话：用户: 经营贷的利率和房贷比哪个更低？\n助手: 目前经营贷政策性利率更低。",
    "最近对话：用户: 我想贷500万，能不能批？\n助手: 大额贷款需要提供更多的收入证明和抵押物。",
    "最近对话：用户: 还款方式怎么选？\n助手: 等额本息每月还款固定，等额本金前期压力大但总利息少。",
    "最近对话：用户: 贷款合同签了，利率还会变吗？\n助手: 如果是固定利率则不变，浮动利率随LPR调整。",
    "最近对话：用户: 提前还贷有违约金吗？\n助手: 满一年通常免收。",
]

TOOL_OPS = [
    "工具操作：助手: query_interest_rate\n工具结果: 首套房利率4.2%",
    "工具操作：助手: calculate_monthly_payment\n工具结果: 月供5300元，总利息90.8万",
    "工具操作：助手: check_loan_eligibility\n工具结果: 基本符合，但征信查询次数较多",
    "工具操作：助手: calculate_max_loan_amount\n工具结果: 最高可贷额度180万",
    "工具操作：助手: compare_loan_products\n工具结果: 等额本金比等额本息节省利息13.2万",
    "工具操作：助手: general_search_knowledge\n工具结果: 申请材料清单已列出",
    "工具操作：助手: calculate_loan_total_cost\n工具结果: 含评估费、保险费等共计约1.2万元",
    "工具操作：助手: generate_repayment_schedule\n工具结果: 已生成36期还款计划表",
    "工具操作：助手: apply_home_loan_skill\n工具结果: 综合评估完成，参考利率4.5%，建议贷款8成",
    "工具操作：助手: apply_consumer_loan_skill\n工具结果: 消费贷额度30万，推荐利率4.8%，月供计算完成",
    "工具操作：助手: query_interest_rate\n工具结果: 二套房利率不低于LPR+60BP，当前为4.95%",
    "工具操作：助手: calculate_monthly_payment\n工具结果: 等额本金首月还款6250元，末月4210元",
    "工具操作：助手: check_loan_eligibility\n工具结果: 因工作年限不足，暂不满足基本准入条件",
    "工具操作：助手: calculate_max_loan_amount\n工具结果: 基于月收入1.5万、零负债，参考额度85万",
    "工具操作：助手: general_search_knowledge\n工具结果: 经营贷申请需提供营业执照满2年",
    "工具操作：助手: query_loan_interest\n工具结果: 您的消费贷意向正在处理中，申请编号LON202606100001",
    "工具操作：助手: upsert_loan_interest\n工具结果: 已帮您登记一笔住房贷款意向，金额200万元，期限30年",
    "工具操作：助手: urge_loan_interest\n工具结果: 已为您的经营贷意向标记为加急处理",
]

# ======================== 工具函数 ========================
def load_chunks(file_path="chunked_docs.jsonl"):
    chunks = []
    if not os.path.exists(file_path):
        print(f"警告：{file_path} 不存在")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def get_random_chunk(chunks: List[Dict]) -> Optional[Dict]:
    if not chunks:
        return None
    return random.choice(chunks)

def build_base_context(with_profile=True, with_summary=True, with_tools=True, with_recent=True) -> str:
    parts = []
    parts.append(f"用户画像：{random.choice(PROFILES) if with_profile else '暂无相关信息'}")
    parts.append(f"对话摘要：{random.choice(SUMMARIES) if with_summary else '暂无相关信息'}")
    parts.append(f"近期工具操作：{random.choice(TOOL_OPS) if with_tools else '暂无相关信息'}")
    parts.append(f"最近对话：{random.choice(RECENT_CONVS) if with_recent else '暂无相关信息'}")
    return "\n".join(parts)

def build_text_a(base_context: str, business_knowledge: str = "暂无相关信息") -> str:
    return f"{base_context}\n业务知识：{business_knowledge}"

def is_valid_sample(text: str) -> bool:
    forbidden = ["我注意到", "根据您提供的信息", "我来提取参数", "目前已知信息"]
    return not any(w in text for w in forbidden) and len(text) >= 2

def call_llm(client, model, prompt, max_tokens=120):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content.strip()
            if result == "SKIP":
                return None
            if result and len(result) >= 2:
                return result
        except Exception as e:
            print(f"LLM 调用失败，重试 {attempt+1}: {e}")
            time.sleep(2)
    return None

# ======================== 严苛提示词模板 ========================
STRICT_DIRECT_PROMPT = """你是一个正在咨询银行贷款的用户。
你的对话上下文如下：
{text_a}

请严格判断：基于以上上下文，能否生成一个用户提问，该提问的答案必须**完整、明确、直接**地从上下文中找到，不能仅凭一两个关键词重叠就认为可以回答。

如果能，请输出这个口语化的用户提问（一句话，不要任何解释）。
如果不能，请只输出 SKIP。

输出规则（严格遵守，违反将被拒绝）：
1. 如果判断为“能”，只输出一个纯净的口语化问句。不要包含“能。”、“可以。”、“提问：”等任何前缀、后缀或解释。
2. 如果判断为“不能”，只输出单词 SKIP，不要包含任何其他文字。
3. 绝对不要输出换行符，你的回复必须是一行文本。
4. 绝对不要输出思考过程，只输出最终结果。

示例：
- 正确输出：您之前提到的那笔贷款，最后批了多少额度？
- 错误输出：能。 根据上下文，提问：您之前提到的那笔贷款，最后批了多少额度？
- 正确输出：SKIP
- 错误输出：不能。因为上下文没有提到具体金额。

现在请输出："""

# ======================== 生成函数 ========================
def generate_samples_business_knowledge(client, model, chunks, count):
    """基于业务知识生成样本，业务知识字段有内容，答案在业务知识中"""
    samples = []
    while len(samples) < count:
        chunk = get_random_chunk(chunks)
        if not chunk:
            break
        content = chunk.get("content", "")
        if not content:
            continue
        # 随机单轮/多轮
        if random.random() < 0.4:
            base = "用户画像：暂无相关信息\n对话摘要：暂无相关信息\n近期工具操作：暂无相关信息\n最近对话：暂无相关信息"
        else:
            base = build_base_context()
        text_a = build_text_a(base, content)
        question = call_llm(client, model, STRICT_DIRECT_PROMPT.format(text_a=text_a))
        if question and is_valid_sample(question):
            samples.append({"text_a": text_a, "text_b": question, "label": "DIRECT_REPLY"})
    return samples

def generate_samples_context(client, model, count, with_business_knowledge=False, chunks=None):
    """基于上下文生成样本，业务知识字段为空或可选，答案在其他字段"""
    samples = []
    while len(samples) < count:
        base = build_base_context(with_tools=True, with_recent=True)
        bk = "暂无相关信息"
        if with_business_knowledge and chunks:
            chunk = get_random_chunk(chunks)
            if chunk:
                bk = chunk.get("content", "")[:600]
        text_a = build_text_a(base, bk)
        question = call_llm(client, model, STRICT_DIRECT_PROMPT.format(text_a=text_a))
        if question and is_valid_sample(question):
            samples.append({"text_a": text_a, "text_b": question, "label": "DIRECT_REPLY"})
    return samples

def generate_samples_noise(client, model, chunks, count):
    """干扰项：有业务知识但与问题无关，答案在上下文其他地方"""
    samples = []
    while len(samples) < count:
        # 获取一个业务知识片段
        chunk = get_random_chunk(chunks)
        if not chunk:
            break
        bk_content = chunk.get("content", "")[:600]
        base = build_base_context(with_tools=True, with_recent=True)
        text_a = build_text_a(base, bk_content)
        # 要求 LLM 生成一个与业务知识无关、但可以从其他上下文回答的问题
        prompt = f"""你是一个正在咨询银行贷款的用户。
你的对话上下文如下：
{text_a}

请严格判断：基于上述上下文中**除“业务知识”以外的部分**（即用户画像、对话摘要、近期工具操作、最近对话），能否生成一个用户提问，其答案必须完整、明确、直接地从中找到？

如果能，请输出这个提问。如果不能，请只输出 SKIP。

输出规则（严格遵守，违反将被拒绝）：
1. 如果判断为“能”，只输出一个纯净的口语化问句。不要包含任何前缀、后缀或解释。
2. 如果判断为“不能”，只输出单词 SKIP，不要包含任何其他文字。
3. 绝对不要输出换行符，你的回复必须是一行文本。
4. 绝对不要输出思考过程，只输出最终结果。
5. 不要从“业务知识”字段中找答案。

现在请输出："""
        question = call_llm(client, model, prompt)
        if question and is_valid_sample(question) and "\n" not in question:
            samples.append({"text_a": text_a, "text_b": question, "label": "DIRECT_REPLY"})
    return samples

# ======================== 主函数 ========================
def main(args):
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    model = args.model
    chunks = load_chunks(args.chunk_file)
    random.seed(RANDOM_SEED)

    bk_count = int(TOTAL_SAMPLES * 0.2)
    context_count = int(TOTAL_SAMPLES * 0.6)
    noise_count = TOTAL_SAMPLES - bk_count - context_count

    all_samples = []
    print("生成基于业务知识的样本...")
    all_samples.extend(generate_samples_business_knowledge(client, model, chunks, bk_count))
    print("生成基于上下文的样本（无业务知识）...")
    all_samples.extend(generate_samples_context(client, model, context_count, with_business_knowledge=False))
    print("生成干扰项样本...")
    all_samples.extend(generate_samples_noise(client, model, chunks, noise_count))

    # 去重
    seen = set()
    unique = []
    for item in all_samples:
        key = (item["text_a"], item["text_b"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    output_file = args.output or "direct_reply_samples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n生成完毕，共 {len(unique)} 条 DIRECT_REPLY 样本，保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--chunk_file", type=str, default="chunked_docs.jsonl")
    parser.add_argument("--output", type=str, default="direct_reply_samples.jsonl")
    args = parser.parse_args()
    main(args)