#!/usr/bin/env python3
"""
具体工具/Skill 训练数据生成脚本（修正版）
包含12个标签：query_interest_rate, calculate_monthly_payment, calculate_max_loan_amount,
calculate_loan_total_cost, check_loan_eligibility, compare_loan_products,
generate_repayment_schedule, apply_home_loan_skill, apply_consumer_loan_skill,
query_loan_interest, upsert_loan_interest, urge_loan_interest

修正点：
1. 丰富上下文模板
2. 增加信息泄漏约束
3. 易混淆标签增加正反例
4. 增加生成后校验
"""

import json
import random
import time
import argparse
import os
from typing import List, Dict, Optional
from openai import OpenAI

# ======================== 配置 ========================
DEFAULT_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SAMPLES_PER_LABEL = 20
SINGLE_TURN_RATIO = 0.6   # 单轮占比
RANDOM_SEED = 42

OUTPUT_FILE = "advisor_tools_samples.jsonl"

# ======================== 完整上下文模板 ========================
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
    parts = []
    parts.append(f"用户画像：{random.choice(PROFILES) if with_profile else '暂无相关信息'}")
    parts.append(f"对话摘要：{random.choice(SUMMARIES) if with_summary else '暂无相关信息'}")
    parts.append(f"近期工具操作：{random.choice(TOOL_OPS) if with_tools else '暂无相关信息'}")
    parts.append(f"最近对话：{random.choice(RECENT_CONVS) if with_recent else '暂无相关信息'}")
    return "\n".join(parts)

def build_text_a(base_context: str) -> str:
    return f"{base_context}\n业务知识：暂无相关信息"

def is_valid_sample(text_b: str) -> bool:
    """简单校验：避免包含推理词汇或过短"""
    forbidden = ["我注意到", "根据您提供的信息", "我来提取参数", "目前已知信息"]
    if any(w in text_b for w in forbidden):
        return False
    if len(text_b) < 4:
        return False
    return True

# ======================== 标签专属 Prompt 定义（已增强约束） ========================
LABEL_PROMPTS = {
    "query_interest_rate": {
        "desc": "查询当前贷款利率",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**查询贷款利率**（想知道当前的利率数值）。

上下文：
{text_a}

要求：
1. 提问必须明确要求知道当前的利率数值（如“房贷利率是多少”、“现在利率多少”）
2. 不要问利率政策或概念（那是知识检索），不要问月供或总成本
3. 生成的问题不能从上下文中直接回答。如果上下文已包含利率信息，请输出 SKIP。
4. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "calculate_monthly_payment": {
        "desc": "计算月供",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**计算月供**（每月还款额）。

上下文：
{text_a}

要求：
1. 提问应涉及具体的贷款金额、期限或还款方式，要求计算每月还款多少。
2. 不要问综合评估（那是Skill），不要问总成本（那是总成本计算）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含月供信息，请输出 SKIP。
4. 正例：“贷100万30年，每月还多少？”
   反例：“总共要还多少钱包括所有费用？” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "calculate_max_loan_amount": {
        "desc": "计算最高可贷额度",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**计算自己最多能贷多少钱**。

上下文：
{text_a}

要求：
1. 提问应表达想知道最高额度（如“最多能贷多少”、“能贷到多少额度”）。
2. 不要问“能否贷款”或“资格”（那是资格预审）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含额度信息，请输出 SKIP。
4. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "calculate_loan_total_cost": {
        "desc": "计算贷款总成本",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**计算贷款的综合总成本**（包含利息和各种费用）。

上下文：
{text_a}

要求：
1. 提问应明确要求计算全部费用、总花费或总成本（如“总共要还多少钱”、“算一下包含所有费用的总成本”）。
2. 不要只问月供或利息（那是月供计算）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含总成本信息，请输出 SKIP。
4. 正例：“算一下贷款30万，3年，总共要花多少钱？”
   反例：“每月还多少？” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "check_loan_eligibility": {
        "desc": "贷款资格预审",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**检查自己是否符合贷款申请条件**。

上下文：
{text_a}

要求：
1. 提问应询问“我能不能贷”、“我是否符合条件”、“我满足要求吗”。
2. 不要问最高额度（那是计算额度）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含资格信息，请输出 SKIP。
4. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "compare_loan_products": {
        "desc": "贷款方案对比",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**对比不同的贷款产品、方案或还款方式**。

上下文：
{text_a}

要求：
1. 提问应明确要求对比（如“哪个更划算”、“有什么区别”、“帮我比较一下”）。
2. 不要只问单一产品的细节（那是知识检索）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含对比结论，请输出 SKIP。
4. 正例：“等额本息和等额本金哪个更合适我？”“抵押贷和信用贷哪个利率低？”
   反例：“等额本息是什么意思？” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "generate_repayment_schedule": {
        "desc": "生成还款计划表",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**生成详细的还款计划表**。

上下文：
{text_a}

要求：
1. 提问应要求查看还款计划、还款表、每期明细（如“帮我出一份还款计划”、“每一期还多少钱的清单”）。
2. 不要只算月供（那是月供计算）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含还款计划，请输出 SKIP。
4. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "apply_home_loan_skill": {
        "desc": "住房贷款综合评估",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**申请住房贷款并进行全面评估**（包括利率、额度、月供、资格等）。

上下文：
{text_a}

要求：
1. 提问应明确提到购房、买房、住房贷款、按揭、首套房等，并希望得到全面评估或申请指导。
2. 不要只问单一事项（如仅利率或仅月供），那是具体工具。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含评估结果，请输出 SKIP。
4. 正例：“我想买首套房，帮我全面评估一下能贷多少、利率多少、每月还多少。”
   反例：“房贷利率是多少？” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "apply_consumer_loan_skill": {
        "desc": "消费贷款综合评估",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**申请消费贷款并进行全面评估**（如装修、旅游、教育、购车等）。

上下文：
{text_a}

要求：
1. 提问应明确提到消费贷、装修贷、旅游贷、教育贷等，并希望得到全面评估。
2. 不要只问单一事项（那是具体工具）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含评估结果，请输出 SKIP。
4. 正例：“我想贷30万装修，帮我算一下利率、月供，看看我符不符合条件。”
   反例：“装修贷利率多少？” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "query_loan_interest": {
        "desc": "查询贷款意向进度",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**查询自己之前登记的贷款意向的处理进度**。

上下文：
{text_a}

要求：
1. 提问应询问“我的贷款申请怎么样了”、“进度如何”、“状态是什么”、“有没有人联系我”。
2. 不要问通用的贷款政策（那是知识检索）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含进度信息，请输出 SKIP。
4. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "upsert_loan_interest": {
        "desc": "提交/更新贷款意向",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**提交新的贷款意向或修改之前的意向金额、期限等**。

上下文：
{text_a}

要求：
1. 提问应表达“帮我提交申请”、“我要登记贷款”、“修改金额/期限”、“重新激活”等操作。
2. 不要问综合评估（那是Skill）；不要只查进度（那是查询意向）。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含提交结果，请输出 SKIP。
4. 正例：“帮我提交贷款申请，30万，5年。”“把之前登记的金额改成50万。”
   反例：“我的贷款申请到哪一步了？” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    },
    "urge_loan_interest": {
        "desc": "催促贷款意向处理",
        "prompt_template": """你是一个正在咨询银行贷款的用户。基于下面的对话上下文，请生成一句口语化的用户提问，意图是**催促加急处理自己的贷款申请**。

上下文：
{text_a}

要求：
1. 提问应明确要求“加急”、“催一下”、“快点处理”、“优先审批”。
2. 不要只是抱怨或情绪发泄（如“太慢了”），那是 DIRECT_REPLY。
3. 生成的问题不能从上下文中直接回答。如果上下文已包含催促结果，请输出 SKIP。
4. 正例：“帮我催一下我的贷款申请，等太久了。”
   反例：“怎么还没批下来，太慢了！” → SKIP
5. 口语化，只输出问题本身。如果无法生成合适的问题，输出 SKIP。"""
    }
}

# ======================== 生成函数 ========================
def generate_label_samples(client, model, label, prompt_template, count):
    samples = []
    attempts = 0
    max_attempts = count * 5
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        # 随机选择单轮或多轮（单轮也可能有少量上下文）
        if random.random() < SINGLE_TURN_RATIO:
            # 单轮：大部分字段为空，但偶尔保留画像
            base_context = build_base_context(
                with_profile=random.random() < 0.2,
                with_summary=False,
                with_tools=False,
                with_recent=False
            )
        else:
            base_context = build_base_context(
                with_profile=random.random() > 0.3,
                with_summary=random.random() > 0.3,
                with_tools=random.random() > 0.5,
                with_recent=random.random() > 0.2
            )
        text_a = build_text_a(base_context)
        prompt = prompt_template.format(text_a=text_a)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=100,
            )
            result = resp.choices[0].message.content.strip()
            if result and result != "SKIP" and is_valid_sample(result):
                samples.append({"text_a": text_a, "text_b": result, "label": label})
                print(f"[{label}] 生成 {len(samples)}/{count}")
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            time.sleep(2)
    return samples

# ======================== 主函数 ========================
def main(args):
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    model = args.model
    random.seed(RANDOM_SEED)

    all_samples = []
    for label, cfg in LABEL_PROMPTS.items():
        print(f"\n开始生成 {label} 样本...")
        samples = generate_label_samples(client, model, label, cfg["prompt_template"], SAMPLES_PER_LABEL)
        all_samples.extend(samples)
        print(f"{label} 实际生成: {len(samples)} 条")

    # 去重
    seen = set()
    unique = []
    for item in all_samples:
        key = (item["text_a"], item["text_b"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # 保存前校验
    print("\n数据校验...")
    label_counts = {}
    for item in unique:
        lbl = item["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        # 确保 business_knowledge 为空
        assert "业务知识：暂无相关信息" in item["text_a"], f"{lbl} 样本 business_knowledge 不为空！"
    print("标签分布：")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl}: {cnt}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n全部生成完毕，共 {len(unique)} 条样本，保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    main(args)