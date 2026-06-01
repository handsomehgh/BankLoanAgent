#!/usr/bin/env python3
"""
离线合成训练数据脚本 —— 修正版（明确区分单轮/多轮样本）

- 单轮样本（text_a = ""）：用户首次开口，无上下文
- 多轮样本（text_a 包含画像/摘要/对话等）：模拟真实对话中的追问或切换
- 标签预定义，100%准确
"""

import json
import random
import time
import re
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter

from openai import OpenAI

# ======================== 配置 ========================
DEFAULT_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"          # 替换为真实 Key
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SAMPLES_PER_LABEL = 150          # 每类标签总样本数
SINGLE_TURN_RATIO = 0.6          # 60% 单轮样本，40% 多轮样本
VAL_RATIO = 0.15
OUTPUT_TRAIN = "train.jsonl"
OUTPUT_VAL = "val.jsonl"
RANDOM_SEED = 42

# ======================== 合法标签 ========================
ALL_LABELS = [
    "DIRECT_REPLY",
    "CLARIFY",
    "apply_home_loan_skill",
    "apply_consumer_loan_skill",
    "calculate_monthly_payment",
    "query_interest_rate",
    "calculate_loan_total_cost",
    "calculate_max_loan_amount",
    "check_loan_eligibility",
    "compare_loan_products",
    "generate_repayment_schedule",
    "general_search_knowledge",
]

# ======================== 扩展的上下文模板库 ========================
# (此处沿用你之前确认过的完整模板库，保持不变)
CONTEXT_TEMPLATES = {
    "profile_only": [
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
    ],
    "summary_only": [
        "对话摘要：用户刚咨询了房贷利率，助手回复首套房LPR 4.2% + 30BP。",
        "对话摘要：用户询问消费贷额度，助手表示需要评估收入和负债。",
        "对话摘要：用户对等额本息和等额本金的概念有疑问，助手已解释。",
        "对话摘要：用户表达了想申请装修贷款的意愿，助手介绍了基本条件。",
        "对话摘要：用户想了解经营贷和消费贷的区别，助手对比了利率和期限。",
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
        "对话摘要：用户想了解线上申请贷款的额度限制，助手答复纯信用消费贷最高30万。",
        "对话摘要：用户询问了助学贷款或留学贷款的可能性，助手介绍了教育贷款产品。",
        "对话摘要：用户询问了以房养老的反向抵押贷款，助手表示暂无此类产品。",
        "对话摘要：用户咨询了车位贷或装修贷的具体利率和期限，助手给出了参考区间。",
        "对话摘要：用户询问了贷款保险的必要性，助手解释房贷通常强制购买抵押物保险。",
    ],
    "recent_conv_only": [
        "最近对话：用户: 那贷100万30年月供呢？\n助手: 请提供年利率和还款方式。",
        "最近对话：用户: 能贷多少？\n助手: 您月收入和现有负债大概多少？",
        "最近对话：用户: 提前还贷有违约金吗？\n助手: 满一年通常免收。",
        "最近对话：用户: 等额本息和等额本金哪个更省钱？\n助手: 等额本金总利息更少。",
        "最近对话：用户: 利率怎么这么高？\n助手: 目前的利率是根据您的征信和产品类型定的，不过我可以再帮您查一下最新优惠。",
        "最近对话：用户: 材料已经提交了，什么时候能审批下来？\n助手: 通常在3个工作日内会有结果，您注意接听电话。",
        "最近对话：用户: 我想贷20万装修，5年还，月供别超过4000行吗？\n助手: 我们试算一下，如果利率按4.5%计算，20万5年月供约为3729元。",
        "最近对话：用户: 我征信有过一次逾期，影响大吗？\n助手: 单次非连续逾期且已结清的话，影响有限，我帮您做下资格预审。",
        "最近对话：用户: 公积金余额能用来付首付吗？\n助手: 公积金一般不能直接用于首付，但可以在贷款下来后提取用来还贷。",
        "最近对话：用户: 我老婆没有工作，我们还能贷款吗？\n助手: 可以用您的收入作为主要还款来源，我们试算一下最高额度。",
        "最近对话：用户: 还款日我忘了，今天才补上，算逾期吗？\n助手: 一般在还款日后3天内都算宽限期，具体看合同约定。",
        "最近对话：用户: 我想把等额本息改成等额本金可以吗？\n助手: 还款方式变更需要申请，可能涉及手续费，我帮您试算一下。",
        "最近对话：用户: 我在外地有套房，再买一套算首套吗？\n助手: 认房又认贷的城市可能算二套，具体要看当地政策。",
        "最近对话：用户: 贷款批下来钱打到哪里？\n助手: 房贷会直接划入开发商或卖方账户，消费贷则打入您的个人账户。",
        "最近对话：用户: 我已经有两张信用卡了，再贷款会不会影响审批？\n助手: 信用卡本身不影响，但已用额度会算入负债，我帮您算一下DTI。",
        "最近对话：用户: 我月收入1万多，但都是现金，没有流水，能贷款吗？\n助手: 您可以提供其他资产证明，比如房产、车辆、存单等。",
        "最近对话：用户: 经营贷的利率和房贷比哪个更低？\n助手: 目前经营贷政策性利率更低，但需要有实际经营的营业执照和流水。",
        "最近对话：用户: 我想贷500万，能不能批？\n助手: 大额贷款需要提供更多的收入证明和抵押物，我们一步步来。",
        "最近对话：用户: 还款方式怎么选？\n助手: 等额本息每月还款固定，等额本金前期压力大但总利息少，您更看重哪方面？",
        "最近对话：用户: 贷款合同签了，利率还会变吗？\n助手: 如果是固定利率则不变，如果是浮动利率则会随LPR调整，重定价周期一年一次。",
    ],
    "tool_ops_only": [
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
        "工具操作：助手: calculate_loan_total_cost\n工具结果: 总融资成本年化约5.2%，含所有附加费用",
        "工具操作：助手: compare_loan_products\n工具结果: 20年与30年方案对比，30年月供低但总利息多付28万",
        "工具操作：助手: generate_repayment_schedule\n工具结果: 已生成120期还款计划，每期本金递减",
        "工具操作：助手: apply_home_loan_skill\n工具结果: 首套房资质符合，建议选择LPR浮动利率",
        "工具操作：助手: apply_consumer_loan_skill\n工具结果: 基于月收入8000元，信用消费贷额度最高8万",
    ],
    "mixed_rich": [
        "用户画像：月入2万，无负债。\n对话摘要：用户咨询了房贷申请条件。\n最近对话：用户: 需要哪些材料？\n助手: 身份证、收入证明、银行流水。",
        "用户画像：月入1.8万，有房贷。\n对话摘要：用户想了解消费贷利率。\n工具操作：助手: query_interest_rate\n工具结果: 消费贷年利率4.5%起。\n最近对话：用户: 那贷20万3年月供呢？\n助手: 请提供还款方式。",
        "用户画像：个体户，年流水150万。\n对话摘要：用户咨询经营贷额度和条件。\n最近对话：用户: 需要营业执照满几年？\n助手: 一般要求满2年。\n工具操作：助手: general_search_knowledge\n工具结果: 经营贷准入材料清单。",
        "用户画像：27岁，程序员，月薪2.5万，无负债。\n对话摘要：用户想计算购房能力。\n最近对话：用户: 首付50万，剩下贷款200万30年行吗？\n助手: 按当前利率4.2%估算，月供约9770元。\n工具操作：助手: calculate_monthly_payment\n工具结果: 月供9770元，总利息151.7万。",
        "用户画像：退休人员，月退休金8000元，有存款100万。\n对话摘要：用户想以房养老但被拒。\n最近对话：用户: 那我能做抵押消费贷吗？\n助手: 年龄可能超标，我先帮您查下资格。\n工具操作：助手: check_loan_eligibility\n工具结果: 因年龄超过60岁，贷款期限受限。",
        "用户画像：宝妈，无收入，配偶月入3万。\n对话摘要：用户想以自己的名义申请消费贷。\n最近对话：用户: 我用老公的收入证明可以吗？\n助手: 不能直接用，但可以让他作为共同借款人。\n工具操作：助手: check_loan_eligibility\n工具结果: 无自主还款来源，建议添加共同借款人。",
        "用户画像：刚毕业，月薪7000元，无资产。\n对话摘要：用户想贷款买二手车。\n最近对话：用户: 我能贷5万3年还吗？\n助手: 让我先查下征信和准入条件。\n工具操作：助手: check_loan_eligibility\n工具结果: 月收入覆盖月供，但需提供工作证明。",
        "用户画像：企业主，年利润200万，征信良好。\n对话摘要：用户对比经营贷和抵押贷。\n最近对话：用户: 我的厂房评估500万，能贷多少？\n助手: 抵押率一般70%，最高350万。\n工具操作：助手: calculate_max_loan_amount\n工具结果: 基于资产评估，最高可贷350万。",
        "用户画像：自由职业，收入不稳定，有房无贷。\n对话摘要：用户想抵押房产贷款装修。\n最近对话：用户: 装修需要40万，十年还清。\n助手: 我来算一下月供和总成本。\n工具操作：助手: calculate_loan_total_cost\n工具结果: 月供4410元，总利息约12.9万。",
        "用户画像：教师，公积金连续缴存10年，余额15万。\n对话摘要：用户想申请公积金贷款买房。\n最近对话：用户: 公积金能贷多少？\n助手: 根据您的缴存基数和余额，最高可贷80万。\n工具操作：助手: calculate_max_loan_amount\n工具结果: 公积金贷款额度80万，商贷可组合。",
        "用户画像：有房贷在还，月供2500元，月入2万。\n对话摘要：用户想申请消费贷旅游。\n最近对话：用户: 再贷10万会不会压力大？\n助手: 我帮您算一下DTI负债率。\n工具操作：助手: calculate_max_loan_amount\n工具结果: 当前DTI 12.5%，新增10万贷款后DTI约28%，仍在安全线内。",
        "用户画像：外卖骑手，无社保，月入1.2万。\n对话摘要：用户想贷款买摩托车。\n最近对话：用户: 我没有工资流水怎么办？\n助手: 您可以用近6个月的微信/支付宝流水作为参考。\n工具操作：助手: check_loan_eligibility\n工具结果: 流水可接受，但需补充居住证明。",
        "用户画像：企业高管，月薪5万，持有公司股票。\n对话摘要：用户想贷款买第三套房。\n最近对话：用户: 第三套房还能贷款吗？\n助手: 大部分城市第三套停贷，但可以办理抵押经营贷。\n工具操作：助手: general_search_knowledge\n工具结果: 第三套房限贷政策查询结果。",
        "用户画像：夫妇合计月入6万，已有两套房贷款均已还清。\n对话摘要：用户想为子女购买婚房。\n最近对话：用户: 算首套还是二套？\n助手: 如果孩子名下无房，可以以他的名义申请首套房贷款。\n工具操作：助手: query_interest_rate\n工具结果: 首套房利率4.2%，二套房4.95%。",
        "用户画像：卡车司机，月入1.5万，现金收入。\n对话摘要：用户想办理车贷。\n最近对话：用户: 首付三成，贷15万3年，利息多少？\n助手: 按当前车贷利率4.5%算，月供4458元。\n工具操作：助手: calculate_monthly_payment\n工具结果: 月供4458元，总利息约10,488元。",
    ]
}

# 扩展助手回复池
ASSISTANT_RESPONSES = {
    "ask_for_params": [
        "请问贷款年利率是多少？",
        "请提供贷款本金和期限。",
        "还款方式是等额本息还是等额本金？",
        "您需要贷多少金额，分几年还？",
        "请问您的贷款用途是什么？",
        "您需要我计算月供还是总成本？",
        "请提供您的月收入和现有负债情况。",
        "您想查询哪种贷款产品的利率？",
        "您是需要对比方案还是看还款计划？",
        "请告诉我您的年龄和工作年限，我帮您做资格预审。",
    ],
    "provide_info": [
        "根据当前LPR，首套房利率约4.2%。",
        "最高可贷额度约150万。",
        "消费贷利率在3.95%-5.45%之间。",
        "申请房贷需要身份证、收入证明、银行流水。",
        "等额本息每月还款固定，等额本金总利息更少。",
        "经营贷需要营业执照满2年，且提供经营流水。",
        "公积金贷款额度受缴存年限和余额影响。",
        "贷款审批通过后，一般1-3个工作日放款。",
        "提前还款满一年通常免收违约金。",
        "征信查询次数过多会影响审批，建议间隔3个月。",
    ],
    "tool_result": [
        "月供5300元，总利息90.8万。",
        "信用评分良好，符合准入条件。",
        "抵押率可达70%，最高额度350万。",
        "当前DTI为28%，在安全范围内。",
        "公积金贷款最高可贷80万。",
        "等额本金比等额本息节省利息13.2万。",
        "二套房利率不低于LPR+60BP，当前为4.95%。",
        "经营贷年利率3.95%起，随LPR浮动。",
        "还款计划已生成，前3期月供包含更多利息。",
        "综合成本包含评估费、保险费等，总计约1.2万元。",
    ],
}

# 更多最近对话起始模板（可动态生成对话）
RECENT_CONV_STARTERS = [
    "用户: {user_input}\n助手: {assistant_response}",
    "用户: {user_input}\n助手: {assistant_response}\n用户: {follow_up}",
]

# ======================== 数据清洗工具 ========================
FORBIDDEN_REASONING_WORDS = [
    "我注意到", "根据您提供的信息", "我来提取参数", "目前已知信息",
    "让我追问", "好的，我来帮您", "根据上下文", "让我重新调用工具",
    "首先，我需要", "我先确认一下", "然后，我会", "基于以上分析",
]

def is_clean_text(text: str) -> bool:
    for word in FORBIDDEN_REASONING_WORDS:
        if word in text:
            return False
    return True

# ======================== 核心生成逻辑 ========================
def build_text_a(rich_context: bool = False) -> str:
    """构造多轮样本的上下文 text_a（与之前相同）"""
    components = []
    if random.random() < 0.6:
        components.append(random.choice(CONTEXT_TEMPLATES["profile_only"]))
    if random.random() < 0.5:
        components.append(random.choice(CONTEXT_TEMPLATES["summary_only"]))
    if random.random() < 0.7:
        components.append(random.choice(CONTEXT_TEMPLATES["recent_conv_only"]))
    if random.random() < 0.3:
        components.append(random.choice(CONTEXT_TEMPLATES["tool_ops_only"]))
    if rich_context and random.random() < 0.4:
        return random.choice(CONTEXT_TEMPLATES["mixed_rich"])
    if not components:
        components.append("暂无相关信息")
    return "\n".join(components)

def generate_single_turn_sample(client, model, label: str, num: int) -> List[Dict]:
    """生成单轮样本：text_a 为空，用户首次开口"""
    samples = []
    for _ in range(num):
        prompt = f"""你是一个正在咨询银行贷款的用户。这是你第一次开口提问，没有任何上下文。
你的提问意图应该被分类为：**{label}**。
请直接输出一句自然的、口语化的提问，不要任何解释。
规则：
- 如果 label 是 DIRECT_REPLY，问题应为常识性、情绪化或确认类。
- 如果 label 是 CLARIFY，问题应极度模糊。
- 如果 label 是某个工具，问题应能体现该工具的意图。
"""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=100,
                )
                text_b = resp.choices[0].message.content.strip()
                if not is_clean_text(text_b) or len(text_b) < 2:
                    continue
                if not re.search(r'[\u4e00-\u9fff]', text_b):
                    continue
                samples.append({
                    "text_a": "",   # 单轮无上下文
                    "text_b": text_b,
                    "label": label
                })
                break
            except Exception as e:
                print(f"单轮生成失败，重试 {attempt+1}: {e}")
                time.sleep(1)
    return samples

def generate_multi_turn_sample(client, model, label: str, num: int) -> List[Dict]:
    """生成多轮样本：text_a 包含丰富上下文"""
    samples = []
    for _ in range(num):
        rich = random.random() < 0.3
        text_a = build_text_a(rich)
        prompt = f"""你是一个正在咨询银行贷款的用户。根据下面的对话上下文，请生成一句**符合逻辑的、自然的用户提问**，使得这个提问的意图恰好应该被分类为：**{label}**。

上下文：
{text_a}

规则：
1. 用户提问必须**简洁、口语化**，只输出问题本身，不要任何解释。
2. 绝对不能输出“我注意到”、“根据上下文”、“我来提取参数”等推理文本。
"""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=100,
                )
                text_b = resp.choices[0].message.content.strip()
                if not is_clean_text(text_b) or len(text_b) < 2:
                    continue
                if not re.search(r'[\u4e00-\u9fff]', text_b):
                    continue
                samples.append({
                    "text_a": text_a[:500],
                    "text_b": text_b,
                    "label": label
                })
                break
            except Exception as e:
                print(f"多轮生成失败，重试 {attempt+1}: {e}")
                time.sleep(1)
    return samples

def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model = args.model or DEFAULT_MODEL
    total_per_label = args.samples or SAMPLES_PER_LABEL
    single_num = int(total_per_label * SINGLE_TURN_RATIO)
    multi_num = total_per_label - single_num

    all_data = []
    for label in ALL_LABELS:
        print(f"生成 {label} 样本 (单轮{single_num} + 多轮{multi_num})...")
        single_data = generate_single_turn_sample(client, model, label, single_num)
        multi_data = generate_multi_turn_sample(client, model, label, multi_num)
        all_data.extend(single_data)
        all_data.extend(multi_data)
        print(f"  单轮{len(single_data)}条, 多轮{len(multi_data)}条")

    # 去重
    seen = set()
    unique = []
    for item in all_data:
        key = (item["text_a"], item["text_b"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    all_data = unique
    print(f"去重后总样本数: {len(all_data)}")

    # 分层划分 train/val
    from collections import defaultdict
    by_label = defaultdict(list)
    for item in all_data:
        by_label[item["label"]].append(item)

    train, val = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        split = max(1, int(len(items) * (1 - VAL_RATIO)))
        train.extend(items[:split])
        val.extend(items[split:])

    random.shuffle(train)
    random.shuffle(val)

    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(out_dir / OUTPUT_VAL, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n训练集: {out_dir/OUTPUT_TRAIN} ({len(train)}条)")
    print(f"验证集: {out_dir/OUTPUT_VAL} ({len(val)}条)")
    print("标签分布(训练集):")
    c = Counter(item["label"] for item in train)
    for lbl in sorted(c):
        print(f"  {lbl}: {c[lbl]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_LABEL, help="每类标签生成样本数")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    args = parser.parse_args()
    random.seed(RANDOM_SEED)
    main(args)