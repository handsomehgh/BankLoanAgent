# author hgh
# version 1.0
# !/usr/bin/env python3
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
from typing import Dict, List, Optional
from collections import Counter

from openai import OpenAI

# ======================== 配置 ========================
DEFAULT_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"  # 替换为真实 Key
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SAMPLES_PER_LABEL = 160  # 每类标签总样本数
SINGLE_TURN_RATIO = 0.6  # 60% 单轮样本，40% 多轮样本
VAL_RATIO = 0.15
OUTPUT_TRAIN = "risk_train.jsonl"
OUTPUT_VAL = "risk_val.jsonl"
RANDOM_SEED = 42

# ======================== 合法标签 ========================
ALL_LABELS = [
    "DIRECT_REPLY",
    "CLARIFY",
    "risk_assessment_skill",
    "calculate_dti",
    "calculate_ltv",
    "calculate_dscr",
    "estimate_credit_score",
    "query_regulation",
    "general_search_knowledge",
]

# ======================== 扩展的上下文模板库 ========================
# (此处沿用你之前确认过的完整模板库，保持不变)
CONTEXT_TEMPLATES = {
    "profile_only": [
        "用户画像：月收入1.2万，有车贷月供2500元，信用卡负债2万，征信记录良好。",
        "用户画像：月收入3.5万，无负债，名下有房产市值约500万，征信优秀。",
        "用户画像：个体经营者，年经营净收入40万，年还本付息额12万，征信有1次逾期已结清。",
        "用户画像：月收入2万，配偶无收入，每月其他债务3000元，欲购二套房。",
        "用户画像：月收入1.8万，现有房贷月供6000元，征信查询次数较多（近半年6次）。",
        "用户画像：自由职业，年收入不稳定，平均月入1.5万，无抵押物，征信白户。",
        "用户画像：退休人员，月退休金7000元，无负债，无征信记录。",
        "用户画像：企业高管，年收入80万，现有经营贷200万，抵押物评估值350万。",
        "用户画像：教师，月收入9000元，公积金缴纳基数1万，名下无房，首次购房。",
        "用户画像：快递员，月收入8000元，有网贷2万，征信近期有2次逾期。",
        "用户画像：企业主，年营业额500万，年净利润80万，现有贷款年还本付息30万。",
        "用户画像：月收入1.5万，名下无负债，但信用卡使用率高（90%），征信良好。",
        "用户画像：海员，年收入25万，房贷月供4000元，因出海有6个月未还记录但已补齐。",
        "用户画像：刚毕业大学生，月薪7000元，无负债，无信用记录。",
        "用户画像：餐厅老板，经营贷100万，抵押物评估值150万，受疫情影响收入减半。",
    ],
    "summary_only": [
        "对话摘要：用户刚咨询了风险评级标准，助手解释DTI≤50%为安全线。",
        "对话摘要：用户担心征信查询次数多会影响贷款，助手解释频繁查询会降低评分。",
        "对话摘要：用户想了解经营贷的偿债覆盖率要求，助手说明一般要求≥1.2。",
        "对话摘要：用户对LTV超限表示困惑，助手解释了抵押率上限政策。",
        "对话摘要：用户询问征信报告中的“连三累六”含义，助手给出了专业解释。",
        "对话摘要：用户想评估自己的综合风险等级，助手开始收集收入和负债信息。",
        "对话摘要：用户表示最近征信被多家银行查询，担心房贷审批。",
        "对话摘要：用户询问抵押物价值下降是否需要补充抵押物，助手解释银行重估规则。",
        "对话摘要：用户想查询个人贷款管理办法的相关规定。",
        "对话摘要：用户担心自己月收入不高但有大量存款，能否获得贷款。",
        "对话摘要：用户咨询了小微企业信用评分标准。",
        "对话摘要：用户想了解征信异议处理流程。",
        "对话摘要：用户提供了收入和负债信息，助手正在进行DTI试算。",
    ],
    "recent_conv_only": [
        # DTI/LTV/DSCR相关
        "最近对话：用户: 我月入2万，车贷每月2500，还能贷多少房贷？\n助手: 我先帮您算一下负债率。",
        "最近对话：用户: 我的房子评估500万，想贷350万，会不会超标？\n助手: 我帮您算LTV抵押率。",
        "最近对话：用户: 我经营贷每年还本付息15万，年净利润40万，够不够？\n助手: 帮您算DSCR偿债覆盖率。",
        # 信用评分相关
        "最近对话：用户: 我有两次逾期记录，信用评分会降多少？\n助手: 我根据您的征信信息估算一下。",
        "最近对话：用户: 我征信白户，是不是评分低？\n助手: 是的，白户一般会扣20分。",
        "最近对话：用户: 我近三个月征信被查了5次，影响大吗？\n助手: 频繁查询会降低评分，我帮您估算。",
        # 综合风险评估
        "最近对话：用户: 帮我全面评估一下我现在的贷款风险。\n助手: 好的，我需要收集您的收入、负债、征信等信息。",
        "最近对话：用户: 我想申请房贷，但不知道风险大不大，能帮我看看吗？\n助手: 当然，我们一步步来评估。",
        # 法规查询
        "最近对话：用户: 个人贷款管理办法有哪些规定？\n助手: 我帮您检索相关法规条文。",
        "最近对话：用户: 征信管理条例里关于不良记录保留多久？\n助手: 我查一下相关法规。",
        # 政策/流程咨询
        "最近对话：用户: 银行对DTI的要求是不是不能超过55%？\n助手: 是的，不同贷款类型有差异。",
        "最近对话：用户: 抵押率是不是首套房和二套房不一样？\n助手: 首套房一般最高80%，二套房70%。",
        "最近对话：用户: 经营贷的DSCR最低要求是多少？\n助手: 一般要求≥1.2，具体看银行政策。",
        # 情绪化表达
        "最近对话：用户: 我感觉自己负债太高了，肯定贷不了款。\n助手: 先别灰心，我们客观评估一下。",
        "最近对话：用户: 征信上有个逾期记录，是不是就彻底没戏了？\n助手: 不一定，还要看逾期程度和还款情况。",
        "最近对话：用户: 我收入挺高的，为什么银行说风险高？\n助手: 可能和您的负债或征信有关，我们分析一下。",
    ],
    "tool_ops_only": [
        "工具操作：助手: calculate_dti\n工具结果: DTI为32%，处于安全区间。",
        "工具操作：助手: calculate_ltv\n工具结果: LTV为75%，未超过首套房80%上限。",
        "工具操作：助手: calculate_dscr\n工具结果: DSCR为1.5，达标。",
        "工具操作：助手: estimate_credit_score\n工具结果: 信用评分720分，评级良好。",
        "工具操作：助手: query_regulation\n工具结果: 个人贷款管理办法第二十条...",
        "工具操作：助手: risk_assessment_skill\n工具结果: 综合评估：DTI安全，LTV适中，信用良好，整体风险较低。",
        "工具操作：助手: calculate_dti\n工具结果: DTI为58%，超过55%上限，建议降低负债。",
        "工具操作：助手: calculate_ltv\n工具结果: LTV为85%，超过二套房70%上限，需提高首付。",
        "工具操作：助手: estimate_credit_score\n工具结果: 因存在严重逾期，信用评分560分，评级较差。",
    ],
    "mixed_rich": [
        "用户画像：月入2万，车贷3000元，信用卡1万。\n对话摘要：用户想评估自身贷款风险。\n最近对话：用户: 我这种情况房贷能批吗？\n助手: 我们先算一下DTI和LTV。",
        "用户画像：个体户，年净利50万，年还贷20万。\n对话摘要：用户询问经营贷偿债能力。\n工具操作：助手: calculate_dscr\n工具结果: DSCR=2.5，非常健康。\n最近对话：用户: 那我能贷更多吗？",
        "用户画像：自由职业，月入1.5万，征信白户。\n对话摘要：用户担心信用不足。\n最近对话：用户: 我没有信用记录，是不是贷不了款？\n助手: 我帮您估算一下信用评分，并查看哪些因素可加分。",
        "用户画像：企业高管，房贷已清，欲购二套房，抵押物估值600万。\n对话摘要：用户想了解二套房风险。\n工具操作：助手: calculate_ltv\n工具结果: LTV70%，刚好达标。\n最近对话：用户: 有办法提高贷款额度吗？",
        "用户画像：有严重逾期记录，但月入5万。\n对话摘要：用户希望评估是否还能贷款。\n最近对话：用户: 我知道自己征信不好，但收入高，能弥补吗？\n助手: 我帮您综合评估各项风险。",
    ],
}

# 扩展助手回复池
ASSISTANT_RESPONSES = {
    "ask_for_params": [
        "请问您的月收入大概多少？",
        "请问您每月需要偿还的其他债务（如车贷、信用卡）是多少？",
        "请问您想申请的贷款金额和抵押物评估价值是多少？",
        "请问您经营贷的年净收入和每年还本付息额是多少？",
        "请问您是否有逾期记录？最近一次逾期是什么时候？",
        "请提供您近3个月的征信查询次数。",
        "请问您名下是否有其他资产可以作为抵押？",
        "请提供您想查询的法规关键词。",
        "请问您的贷款用途是什么？是住房还是经营？",
    ],
    "provide_info": [
        "您的负债率(DTI)为38%，处于安全区间，贷款审批通过可能性较高。",
        "您的抵押率(LTV)为82%，已超过首套房80%的上限，建议提高首付比例。",
        "您的偿债覆盖率(DSCR)为1.8，远高于1.2的底线要求，还款能力很强。",
        "根据您的征信记录，估算信用评分为680分，属于良好水平。",
        "查询到个人贷款管理办法规定：借款人需具备稳定的收入来源和良好的信用记录。",
        "征信白户会扣除20分，建议先申请一张信用卡建立信用记录。",
        "频繁的征信查询会降低信用评分，建议3个月内不再申请其他贷款。",
    ],
    "tool_result": [
        "DTI计算完成：月总债务8500元，月收入2万，DTI为42.5%，安全。",
        "LTV计算完成：贷款金额350万，抵押物价值500万，LTV为70%，未超标。",
        "DSCR计算完成：年经营净收入60万，年还本付息额40万，DSCR为1.5，达标。",
        "信用评分估算：基础分750，逾期扣分20，查询过多扣分15，最终715分，良好。",
        "法规检索结果：征信逾期记录自还清之日起保留5年。",
        "综合风险评估报告：DTI、LTV、DSCR均在安全线内，信用良好，整体风险低。",
    ],
}

# 更多最近对话起始模板（可动态生成对话）
RECENT_CONV_STARTERS = [
    "用户: {user_input}\n助手: {assistant_response}",
    "用户: {user_input}\n助手: {assistant_response}\n用户: {follow_up}",
    "用户: {user_input}\n助手: {assistant_response}\n用户: {follow_up}\n助手: {follow_up_response}",
]

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
        # 根据标签构建更精细的指令
        special_instructions = ""
        if label == "DIRECT_REPLY":
            special_instructions = (
                "你是一个正在咨询的客户，你的问题是常识性的、情绪化的、确认性的，或者是对已提供信息的简单追问，"
                "不需要银行进行任何计算或检索就能直接回答。"
                "例如：'提前还款划算吗？'（常识）'月供太高了我快还不起了'（情绪）'那你的意思是建议我缩短期限？'（确认）。"
                "注意：一定不要提出需要查询政策、计算利息或生成证明等需要工具的问题。"
            )
        elif label == "CLARIFY":
            special_instructions = (
                "你的提问必须非常模糊，完全无法让银行判断你想办理什么具体业务。"
                "例如：'帮我看看'、'这个怎么弄'、'能办吗'。"
                "一定不要提及任何具体业务关键词（如利率、还款、展期、逾期等）。"
            )
        elif label == "extension_management_skill":
            special_instructions = (
                "你想申请或了解贷款展期的整体事宜，可能需要银行先查资格再算方案。"
                "提问应体现你希望银行帮你处理展期问题，而不仅仅是询问资格。"
                "例如：'帮我看看能不能办展期'、'我想申请展期，帮我算算月供能少多少'。"
                "注意：不要只问'我能展期吗'（那属于check_extension_eligibility），"
                "而是要表达出希望银行帮你完成整个展期申请和试算的意愿。"
            )
        elif label == "check_extension_eligibility":
            special_instructions = (
                "你只想知道自己是否符合展期条件，不需要试算或办理。"
                "提问应聚焦在资格检查上，例如：'我能申请展期吗'、'展期需要什么条件'、'我有逾期还能展期吗'。"
                "一定不要提及计算月供或后续办理，否则会混淆成extension_management_skill。"
            )
        elif label == "general_search_knowledge":
            special_instructions = (
                "你想了解贷后相关的政策、流程、材料等非计算类信息。"
                "例如：'解押需要哪些材料'、'贷后检查会查什么'、'逾期记录多久消除'。"
                "一定不要提出需要计算利息、月供、违约金等具体数值的问题。"
            )
        else:
            special_instructions = (
                "你的提问意图应能直接对应到贷款工具：{label}。"
                "请直接输出一句自然的、口语化的提问，不要任何解释。"
            )

        # 构建完整prompt
        prompt = f"""你是一个正在咨询银行贷款的用户。这是你第一次开口提问，没有任何上下文。
    你的提问意图应该被分类为：**{label}**。
    {special_instructions}
    请直接输出一句简洁、口语化的提问，不要任何解释。
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
                    "text_a": "",  # 单轮无上下文
                    "text_b": text_b,
                    "label": label
                })
                break
            except Exception as e:
                print(f"单轮生成失败，重试 {attempt + 1}: {e}")
                time.sleep(1)
    return samples


def generate_multi_turn_sample(client, model, label: str, num: int) -> List[Dict]:
    """生成多轮样本：text_a 包含丰富上下文"""
    samples = []
    for _ in range(num):
        rich = random.random() < 0.3
        text_a = build_text_a(rich)

        # 同单轮一样，构造特殊指令
        special_instructions = ""
        if label == "DIRECT_REPLY":
            special_instructions = (
                "结合上下文，你的提问应该是常识性的确认、情绪表达或对助手回复的简单回应，"
                "不需要调用任何工具即可回答。"
            )
        elif label == "CLARIFY":
            special_instructions = (
                "结合上下文，你的提问必须非常模糊，无法推断具体业务。"
            )
        elif label == "extension_management_skill":
            special_instructions = (
                "你的提问应体现希望银行全面处理展期事宜，可能包含资格检查和方案试算。"
                "不要只问资格，要体现'帮我办展期'的综合意图。"
            )
        elif label == "check_extension_eligibility":
            special_instructions = (
                "你的提问仅限于检查展期资格，不要要求试算或办理。"
            )
        elif label == "general_search_knowledge":
            special_instructions = (
                "你的提问只涉及贷后政策、流程、材料查询，不涉及任何数值计算。"
            )
        else:
            special_instructions = (
                "你的提问意图应能直接对应到贷款工具：{label}。"
            )

        prompt = f"""你是一个正在咨询银行贷款的用户。根据下面的对话上下文，请生成一句**符合逻辑的、自然的用户提问**，使得这个提问的意图恰好应该被分类为：**{label}**。

    上下文：
    {text_a}

    规则：
    1. 用户提问必须**简洁、口语化**，只输出问题本身，不要任何解释。
    2. {special_instructions}
    3. 绝对不能输出“我注意到”、“根据上下文”、“我来提取参数”等推理文本。
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
                print(f"多轮生成失败，重试 {attempt + 1}: {e}")
                time.sleep(1)
    return samples


def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model = args.model or DEFAULT_MODEL
    total_per_label = args.samples or SAMPLES_PER_LABEL
    single_num = int(total_per_label * SINGLE_TURN_RATIO)
    multi_num = total_per_label - single_num

    base_samples = args.samples or SAMPLES_PER_LABEL

    # 为易混淆或低分标签单独增加样本量
    label_sample_override = {
        "DIRECT_REPLY": 250,
        "CLARIFY": 180,
        "general_search_knowledge": 200,
        "query_regulation": 180,
    }

    all_data = []
    for label in ALL_LABELS:
        # 获取当前标签应生成的样本总数
        total_per_label = label_sample_override.get(label, base_samples)
        single_num = int(total_per_label * SINGLE_TURN_RATIO)
        multi_num = total_per_label - single_num

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

    print(f"\n训练集: {out_dir / OUTPUT_TRAIN} ({len(train)}条)")
    print(f"验证集: {out_dir / OUTPUT_VAL} ({len(val)}条)")
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
