# author hgh
# version 1.0

#!/usr/bin/env python3
"""
CLARIFY 训练数据生成脚本

定位：BERT 分类器的最后一层兜底（层级4）
- 不能从上下文中直接回答（非 DIRECT_REPLY）
- 不能匹配到具体工具/Skill
- 也不适合去检索知识库（意图本身不明确）
→ 选择 CLARIFY，生成一句追问引导用户澄清意图。

生成策略：
  单轮（约60条）：上下文全空，LLM 直接生成极度模糊、无法判断意图的用户提问。
  多轮（约90条）：提供上下文（画像、摘要、近期工具操作、最近对话），
    但用户问题依然无法被理解。关键：上下文可能包含信息，但问题本身模糊到无法定位意图。

关键：text_a 的 business_knowledge 字段永远为空。
输出：clarify_samples.jsonl
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
SINGLE_TURN_COUNT = 10
MULTI_TURN_COUNT = 15

RANDOM_SEED = 42

# ======================== 多轮上下文模板（复用之前丰富的模板） ========================
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

def build_base_context(with_profile=True, with_summary=True, with_tools=True, with_recent=True) -> str:
    """构造不含业务知识的上下文（四个字段），用于多轮样本"""
    parts = []
    parts.append(f"用户画像：{random.choice(PROFILES) if with_profile else '暂无相关信息'}")
    parts.append(f"对话摘要：{random.choice(SUMMARIES) if with_summary else '暂无相关信息'}")
    parts.append(f"近期工具操作：{random.choice(TOOL_OPS) if with_tools else '暂无相关信息'}")
    parts.append(f"最近对话：{random.choice(RECENT_CONVS) if with_recent else '暂无相关信息'}")
    return "\n".join(parts)

def build_text_a(base_context: str, business_knowledge: str = "暂无相关信息") -> str:
    """拼接完整 text_a（五个字段），CLARIFY 的业务知识永远为空"""
    return f"{base_context}\n业务知识：{business_knowledge}"

def is_clarify_candidate(text: str) -> bool:
    """简单过滤：排除包含明显业务关键词的生成结果"""
    # 如果包含了具体工具的名称或明确的业务量词，说明不够模糊
    forbidden = ["利率", "月供", "额度", "贷多少", "算一下", "申请", "提交", "查询",
                 "意向", "进度", "催一下", "加急", "材料", "流程", "条件", "年限", "期限"]
    text_lower = text.lower()
    return not any(kw in text_lower for kw in forbidden)

def generate_single_turn(client: OpenAI, model: str, count: int) -> List[Dict]:
    """生成单轮 CLARIFY 样本（上下文全空）"""
    samples = []
    attempts = 0
    max_attempts = count * 4  # 允许较多的重试
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        prompt = """你是一个正在咨询银行贷款的用户，但你完全没想好要问什么，只说了一句非常模糊的话。

请生成一句口语化的、极度模糊的用户提问，要求：
1. 不能让银行助手判断出你的具体意图
2. 不能包含任何业务关键词（如贷款、利率、还款、申请、审批、额度）
3. 不能超过10个字
4. 示例：“帮我看一下”、“这个怎么弄”、“那个呢”、“能办吗”

只输出这句话，不要任何解释。"""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=30,
            )
            result = resp.choices[0].message.content.strip()
            if result and len(result) >= 2 and is_clarify_candidate(result):
                base_context = "用户画像：暂无相关信息\n对话摘要：暂无相关信息\n近期工具操作：暂无相关信息\n最近对话：暂无相关信息"
                text_a = build_text_a(base_context)
                samples.append({"text_a": text_a, "text_b": result, "label": "CLARIFY"})
                print(f"单轮 生成 {len(samples)}/{count}: {result}")
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            time.sleep(1)
    return samples

def generate_multi_turn(client: OpenAI, model: str, count: int) -> List[Dict]:
    """生成多轮 CLARIFY 样本（有上下文但问题依然模糊）"""
    samples = []
    attempts = 0
    max_attempts = count * 4
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        # 构造丰富的上下文
        base_context = build_base_context(
            with_profile=random.random() > 0.3,
            with_summary=random.random() > 0.3,
            with_tools=random.random() > 0.5,
            with_recent=True  # 多轮样本至少要有最近对话
        )
        text_a_without_bk = f"{base_context}\n业务知识：暂无相关信息"

        prompt = f"""你是一个正在咨询银行贷款的用户，但你在当前对话中表现得犹豫不决，说了一句非常模糊的话。

对话上下文：
{text_a_without_bk}

请生成一句口语化的、极度模糊的用户提问，要求：
1. 即使结合上下文，银行助手也无法判断你的具体意图
2. 不能包含任何明确的业务关键词（如利率、月供、额度、贷款类型、申请、提交等）
3. 不能超过15个字
4. 不要提出任何具体需求

只输出这句话，不要任何解释。"""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=30,
            )
            result = resp.choices[0].message.content.strip()
            if result and len(result) >= 2 and is_clarify_candidate(result):
                text_a = build_text_a(base_context)
                samples.append({"text_a": text_a, "text_b": result, "label": "CLARIFY"})
                print(f"多轮 生成 {len(samples)}/{count}: {result}")
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            time.sleep(1)
    return samples

# ======================== 主函数 ========================
def main(args):
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    model = args.model
    random.seed(RANDOM_SEED)

    print("生成单轮 CLARIFY 样本...")
    single = generate_single_turn(client, model, SINGLE_TURN_COUNT)

    print("生成多轮 CLARIFY 样本...")
    multi = generate_multi_turn(client, model, MULTI_TURN_COUNT)

    all_samples = single + multi

    # 去重
    seen = set()
    unique = []
    for item in all_samples:
        key = (item["text_a"], item["text_b"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    output_file = args.output or "clarify_samples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n生成完毕，共 {len(unique)} 条 CLARIFY 样本，保存至 {output_file}")
    print(f"  单轮: {len(single)}")
    print(f"  多轮: {len(multi)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default="advisor_clarify_samples.jsonl")
    args = parser.parse_args()
    main(args)