# author hgh
# version 1.0
#!/usr/bin/env python3
"""
预处理阶段 BERT 二分类器训练数据生成脚本
标签：COMPLETE（语义完整）、NEED_CONTEXT（需要上下文增强）
生成训练集和验证集
"""

import json
import random
import time
import argparse
import os
from typing import Optional, List, Dict
from openai import OpenAI

# ======================== 配置 ========================
DEFAULT_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

RANDOM_SEED = 42
DEFAULT_VAL_RATIO = 0.15

COUNTS = {
    "both_empty": 30,
    "only_recent": {"need": 40, "complete": 20},
    "only_summary": {"need": 40, "complete": 20},
    "both": {"need": 30, "complete": 30}
}
OUTPUT_TRAIN = "context_train1.jsonl"
OUTPUT_VAL = "context_val.jsonl"

# ======================== 上下文模板 ========================
SUMMARIES = [
    # ==================== 利率相关 (15条) ====================
    "用户之前咨询了住房贷款利率，助手回复首套房LPR 4.2% + 30BP。",
    "用户询问了消费贷利率，助手告知年化利率约3.95%-5.45%。",
    "用户了解了经营贷和房贷的利率差异，助手解释了政策优惠。",
    "用户问LPR是什么意思，助手解释了贷款市场报价利率的概念。",
    "用户询问首套房和二套房利率差多少，助手回复二套房不低于LPR+60BP。",
    "用户问公积金贷款利率和商贷差多少，助手回复公积金利率更低。",
    "用户问现在利率是不是历史最低，助手回复目前处于较低水平。",
    "用户询问LPR调整后月供会不会变，助手解释了重定价周期。",
    "用户问固定利率和浮动利率的区别，助手建议根据市场走势选择。",
    "用户想了解助学贷款利率，助手介绍了政策优惠利率。",
    "用户询问经营贷利率优惠政策，助手说明优质客户可享受普惠利率。",
    "用户咨询了车贷利率，助手回复3年期车贷利率约4.5%起。",
    "用户问装修贷利率，助手告知与消费贷利率相近。",
    "用户问信用卡分期和贷款利率哪个高，助手对比了两者。",
    "用户询问利率打折活动，助手说明需关注银行最新优惠。",

    # ==================== 额度相关 (15条) ====================
    "用户询问消费贷额度，助手表示需要评估收入和负债。",
    "用户问最高能贷多少，助手回复需要了解收入和负债情况。",
    "用户提供了月收入2万，助手正在核算可贷额度。",
    "用户问公积金余额能否提高贷款额度，助手解释了计算规则。",
    "用户问夫妻双方收入能否合并计算贷款额度，助手说明可以添加共同借款人。",
    "用户问能否用父母收入共同借款，助手说明可以添加共同借款人。",
    "用户问月薪5000能贷多少钱，助手回复需要看贷款类型和期限。",
    "用户表示收入是现金发放，助手询问是否有其他收入证明。",
    "用户询问抵押经营贷的额度，助手表示需根据抵押物评估值计算。",
    "用户问信用贷和抵押贷额度差异，助手说明抵押贷额度更高。",
    "用户咨询了组合贷款额度分配，助手解释公积金和商贷比例。",
    "用户询问已有房贷再贷消费贷的额度限制，助手说明DTI要求。",
    "用户问企业贷款额度如何核定，助手表示需看经营流水和纳税。",
    "用户询问最高贷款成数，助手说明首套房最高80%。",
    "用户问自己能贷多少，助手引导提供月收入和负债信息。",

    # ==================== 申请条件/资格预审 (15条) ====================
    "用户想了解房贷申请条件，助手列出了基本材料清单。",
    "用户询问年龄限制，助手说明贷款到期时年龄不超过70岁。",
    "用户问工作年限要求，助手回复消费贷需在当前工作满6个月。",
    "用户问征信白户能不能贷款，助手回复可能需要提供更多收入证明。",
    "用户问有逾期记录能否贷款，助手说明需看逾期严重程度。",
    "用户问刚换工作能贷款吗，助手询问试用期是否已过。",
    "用户问个体工商户需要什么条件，助手说明需营业执照满2年。",
    "用户问退休人员能否贷款，助手说明年龄限制会影响贷款期限。",
    "用户问无抵押能贷款吗，助手介绍了信用贷款产品。",
    "用户问外地户口能在本地贷款吗，助手表示需看当地政策。",
    "用户问学生能否贷款，助手说明需要稳定收入来源。",
    "用户问企业贷款需要哪些资质，助手列举了营业执照、纳税证明等。",
    "用户问信用卡逾期记录会影响房贷审批吗，助手表示会参考。",
    "用户问网贷记录多会不会影响银行贷款，助手说明频繁查询有影响。",
    "用户询问担保贷款的条件，助手说明需担保人征信良好。",

    # ==================== 还款方式/月供 (10条) ====================
    "用户咨询了等额本息的计算方式，助手解释了公式。",
    "用户问等额本息和等额本金的区别，助手解释了两种还款方式。",
    "用户表示等额本金前期压力太大，助手建议可以考虑等额本息。",
    "用户问等额本息前期还的是不是全是利息，助手解释了构成。",
    "用户问月供会不会随着LPR变化，助手解释浮动利率机制。",
    "用户表示对总利息感到惊讶，助手解释等额本息前期利息占比高。",
    "用户问能不能选双周供，助手介绍了双周供的特点。",
    "用户询问先息后本还款方式，助手说明适合短期周转。",
    "用户问提前还款后月供怎么变，助手说明缩短期限或减少月供。",
    "用户询问还款日遇到节假日怎么办，助手回复顺延至下一个工作日。",

    # ==================== 提前还款/展期/逾期 (10条) ====================
    "用户咨询了提前还款政策，助手说明满一年免收违约金。",
    "用户问提前还一部分本金月供会变吗，助手说明可选缩短期限或减少月供。",
    "用户问提前还款划算还是继续还贷划算，助手建议进行试算对比。",
    "用户询问展期申请条件，助手表示需要评估逾期记录和已还期数。",
    "用户问展期后利率会变吗，助手确认利率将上浮10个基点。",
    "用户问逾期一天会不会上征信，助手说明一般有宽限期。",
    "用户问还款日忘记还款怎么办，助手建议尽快补还。",
    "用户问逾期罚息怎么算，助手解释按合同利率1.5倍计算。",
    "用户问逾期记录何时消除，助手说明还清后保留5年。",
    "用户问展期被拒怎么办，助手建议考虑变更还款方式。",

    # ==================== 材料/流程/政策 (15条) ====================
    "用户问申请房贷需要哪些材料，助手列举了身份证、收入证明、银行流水等。",
    "用户问贷款审批一般需要多久，助手回复信用贷1-3天、房贷1-2周。",
    "用户问面签需要带什么材料，助手说明需要身份证、户口本等原件。",
    "用户问线上能不能申请贷款，助手回复部分产品支持线上申请。",
    "用户问审批通过后多久放款，助手回复抵押登记完成后1-3个工作日。",
    "用户问贷款审批主要看什么，助手说明主要看征信、收入和负债率。",
    "用户问银行流水不够怎么办，助手建议提供其他资产证明。",
    "用户问收入证明怎么开，助手说明需要单位盖章并注明收入金额。",
    "用户问贷款结清后需要办什么手续，助手说明需要办理解除抵押。",
    "用户问贷款合同丢了怎么办，助手回复可以到经办支行补办。",
    "用户问还款卡丢了怎么换卡，助手说明需要到柜台办理还款账户变更。",
    "用户问解押需要哪些材料，助手回复结清证明、身份证、他项权证。",
    "用户问贷后检查会查什么，助手说明主要检查贷款用途和抵押物状况。",
    "用户问续贷需要重新审批吗，助手表示需要重新评估征信和收入。",
    "用户问公积金冲还贷怎么办理，助手讲解月冲和年冲区别。",

    # ==================== 补充场景 (20条) ====================
    "用户询问了贷款用途凭证的保留要求，助手说明需保留发票和合同。",
    "用户想了解抵押物被拆迁后贷款如何处理，助手解释需提前还款或更换抵押物。",
    "用户询问贷款期间能否出售抵押房产，助手说明需先还清贷款解除抵押。",
    "用户表示配偶去世，咨询贷款继承和还款责任问题。",
    "用户想了解因疫情导致的逾期能否申请征信修复，助手说明可提供证明材料。",
    "用户询问了贷款保险的必要性，助手解释房贷通常强制购买抵押物保险。",
    "用户询问贷款期间利率调整，浮动利率贷款的重定价日一般为每年1月1日。",
    "用户想了解提前还款是否会影响个人征信，助手说明正常提前还款对征信无负面影响。",
    "用户咨询贷款合同遗失后如何补办，助手告知可到经办行申请复印件。",
    "用户询问还款卡丢失后如何变更还款账户，助手说明需本人携带新卡和身份证到柜台。",
    "用户想了解贷款期间能否增加共同借款人，助手说明需要重新审批。",
    "用户咨询了房贷转按揭到其他银行的流程和费用。",
    "用户询问续贷和展期的区别，助手解释续贷是重新申请一笔新贷款。",
    "用户想了解抵押物价值下降是否触发银行要求补充抵押物。",
    "用户询问了贷款期间收入下降能否申请降低月供。",
    "用户想了解贷款审批没通过，多久能再申请，助手建议3-6个月后。",
    "用户询问了消费贷资金不得流入楼市的具体监管规定。",
    "用户想了解车贷和装修贷能否同时申请。",
    "用户询问了抵押物评估费的承担方，助手说明一般由申请人承担。",
    "用户询问了贷款用途的限制，助手明确禁止用于购房首付或投资。",
]

RECENT_CONVS = [
    # ==================== 利率/额度 (15条) ====================
    "用户: 那房贷利率现在多少？\n助手: 首套房5年以上LPR 4.2%，加30BP后4.5%。",
    "用户: 能贷多少？\n助手: 需要了解您的月收入和现有负债情况。",
    "用户: 我月入2万，没有负债，最高能贷多少？\n助手: 我帮您算一下最高可贷额度。",
    "用户: 消费贷利率能低到多少？\n助手: 目前最低3.95%起，看您的资质。",
    "用户: 二套房利率高多少？\n助手: 不低于LPR+60BP，当前约4.95%。",
    "用户: 我公积金余额5万，能贷多少？\n助手: 公积金贷款额度受缴存年限和余额影响。",
    "用户: 利率是固定的还是浮动的？\n助手: 您可以自行选择，浮动利率随LPR调整。",
    "用户: 经营贷利率比房贷低吗？\n助手: 目前经营贷政策性利率更低，但需要有营业执照。",
    "用户: 我征信有点花，利率会高吗？\n助手: 征信情况会影响利率定价。",
    "用户: 车贷利率现在多少？\n助手: 3年期车贷利率约4.5%起，具体看车型和首付。",
    "用户: 夫妻共同贷款额度怎么算？\n助手: 可以合并收入，但负债也会合并计算。",
    "用户: 信用贷最高能贷多少？\n助手: 纯信用消费贷一般最高30万。",
    "用户: 抵押贷能贷几成？\n助手: 住宅一般可贷评估值的70%-80%。",
    "用户: 我月薪5000能贷多少钱？\n助手: 需要看贷款类型和期限，我帮您试算。",
    "用户: 你们银行贷款利率有优惠吗？\n助手: 优质客户和首套房可享受优惠利率。",

    # ==================== 还款/月供 (15条) ====================
    "用户: 那贷100万30年月供呢？\n助手: 请提供年利率和还款方式。",
    "用户: 等额本息和等额本金哪个划算？\n助手: 等额本金总利息更少，但前期压力大。",
    "用户: 月供太高了，能降低吗？\n助手: 可以考虑延长贷款期限或变更还款方式。",
    "用户: 贷款批下来钱打到哪里？\n助手: 房贷会直接划入开发商或卖方账户。",
    "用户: 还款日能不能改？\n助手: 可以申请调整，一般每年可改一次。",
    "用户: 月供里包含保险吗？\n助手: 不包含，保险费是单独缴纳的。",
    "用户: 先息后本和等额本息哪个好？\n助手: 先息后本适合短期周转，等额本息适合长期贷款。",
    "用户: 等额本息每月还款额一样吗？\n助手: 是的，每月还款金额固定，便于预算。",
    "用户: 我可以选双周供吗？\n助手: 部分产品支持双周供，可以节省利息。",
    "用户: 月供里本金和利息怎么分的？\n助手: 等额本息前期利息占比高，后期本金占比高。",
    "用户: 提前还一部分后月供怎么变？\n助手: 可以选择缩短期限或减少月供。",
    "用户: 公积金和商贷月供一起扣吗？\n助手: 分开扣款，公积金先扣，余额不足再扣商贷。",
    "用户: 还款日遇到周末怎么办？\n助手: 顺延至下一个工作日扣款。",
    "用户: 等额本金第几年还清最划算？\n助手: 前期利息高，越早还越省利息。",
    "用户: 月供里包括物业费吗？\n助手: 不包括，物业费是单独缴纳的。",

    # ==================== 提前还款/展期/逾期 (15条) ====================
    "用户: 提前还贷有违约金吗？\n助手: 满一年通常免收。",
    "用户: 提前还20万能省多少利息？\n助手: 请提供剩余本金、年利率和已还期数。",
    "用户: 我想一次性还清，需要什么手续？\n助手: 请提供贷款编号，我帮您算一下应还总额。",
    "用户: 逾期3天罚息怎么算？\n助手: 请提供逾期本金和合同年利率。",
    "用户: 我忘了还款，现在补上还来得及吗？\n助手: 一般在3天宽限期内不算逾期。",
    "用户: 我能申请展期吗？\n助手: 请问已还了多少期？有无逾期记录？",
    "用户: 展期后月供能少多少？\n助手: 我帮您试算一下，请补充当前剩余本金和利率。",
    "用户: 缩短期限和减少月供哪个划算？\n助手: 缩短期限节省利息更多，减少月供减轻压力。",
    "用户: 逾期会影响征信吗？\n助手: 一旦上报征信就会留下记录，建议尽快还款。",
    "用户: 展期后利率会变吗？\n助手: 一般会适度上浮，我帮您具体算一下。",
    "用户: 提前还款预约需要多久？\n助手: 一般需要提前1个月向经办行申请。",
    "用户: 我有过逾期还能展期吗？\n助手: 需要看逾期次数和当前状态。",
    "用户: 罚息比正常利息高多少？\n助手: 一般按合同利率的1.5倍计算。",
    "用户: 提前还款后征信会变好吗？\n助手: 正常提前还款对征信无负面影响。",
    "用户: 逾期记录什么时候能消除？\n助手: 从还清那天算起，5年后自动删除。",

    # ==================== 材料/流程/申请 (20条) ====================
    "用户: 我需要准备哪些材料？\n助手: 身份证、收入证明、银行流水。",
    "用户: 贷款审批要多久？\n助手: 信用贷1-3天，房贷约1-2周。",
    "用户: 面签需要带什么？\n助手: 身份证、户口本、结婚证等原件。",
    "用户: 银行流水不够怎么办？\n助手: 可以提供其他资产证明，如房产、存单。",
    "用户: 收入证明怎么开？\n助手: 需要单位盖章并注明月收入金额。",
    "用户: 审批通过后多久放款？\n助手: 抵押登记完成后1-3个工作日。",
    "用户: 贷款合同丢了怎么办？\n助手: 可以到经办支行申请复印件。",
    "用户: 解押需要哪些材料？\n助手: 结清证明、身份证、他项权证。",
    "用户: 还款卡丢了怎么变更？\n助手: 带新卡和身份证到柜台办理。",
    "用户: 公积金冲还贷怎么办理？\n助手: 到公积金中心或手机APP签约办理。",
    "用户: 贷后检查会查什么？\n助手: 主要检查贷款用途是否合规、抵押物状况。",
    "用户: 续贷需要重新审批吗？\n助手: 需要重新评估您的征信和收入情况。",
    "用户: 贷款期间能卖房吗？\n助手: 需要先还清贷款解除抵押才能过户。",
    "用户: 抵押物被拆迁了怎么办？\n助手: 需要用拆迁款提前还款或更换抵押物。",
    "用户: 贷款用途凭证怎么准备？\n助手: 保留好发票和合同，按要求上传。",
    "用户: 房贷转按揭怎么操作？\n助手: 需要向新银行申请，重新评估审批。",
    "用户: 贷款保险必须买吗？\n助手: 房贷通常需要购买抵押物财产保险。",
    "用户: 线上能申请贷款吗？\n助手: 纯信用消费贷可全程手机银行办理。",
    "用户: 审批没通过还能再申请吗？\n助手: 建议改善征信和负债后3个月再试。",
    "用户: 贷款下来后能直接取现吗？\n助手: 消费贷会打入您账户，可以取用。",

    # ==================== 产品对比/其他 (15条) ====================
    "用户: 房贷和消费贷有什么区别？\n助手: 房贷用于买房，利率低、期限长。",
    "用户: 抵押贷和信用贷哪个好？\n助手: 抵押贷利率低额度高，信用贷手续简便放款快。",
    "用户: 公积金贷款和商贷哪个划算？\n助手: 公积金利率更低，但额度有限。",
    "用户: 经营贷需要什么条件？\n助手: 营业执照满2年、经营流水、纳税证明。",
    "用户: 装修贷和消费贷哪个利率低？\n助手: 利率相近，装修贷有特定用途限制。",
    "用户: 助学贷款怎么申请？\n助手: 需提供录取通知书和家庭经济困难证明。",
    "用户: 车位贷能贷多少？\n助手: 一般不超过车位评估值的70%。",
    "用户: 信用卡分期和贷款哪个划算？\n助手: 贷款通常利率更低，分期手续费较高。",
    "用户: 网贷和银行贷款有什么区别？\n助手: 银行利率更低、更安全。",
    "用户: 组合贷款公积金和商贷比例怎么定？\n助手: 公积金部分按额度上限，剩余用商贷补齐。",
    "用户: 我能用父母的公积金贷款吗？\n助手: 父母可以作为共同借款人使用公积金。",
    "用户: 贷款下来后还能追加额度吗？\n助手: 需要重新审批，不能直接追加。",
    "用户: 房贷批了还能改还款方式吗？\n助手: 放款前可以修改，放款后需要申请变更。",
    "用户: 贷款期间利率会变吗？\n助手: 浮动利率会随LPR调整，固定利率不变。",
    "用户: 征信不好能贷款吗？\n助手: 可以尝试申请，但可能影响额度和利率。",
]

# ======================== 工具函数 ========================
def format_text_a(summaries: Optional[str] = None, recent: Optional[str] = None) -> str:
    sum_part = summaries if summaries else "暂无相关信息"
    rec_part = recent if recent else "暂无相关信息"
    return f"相关对话历史：\n{sum_part}\n最近对话：\n{rec_part}"

def is_valid(text: str) -> bool:
    forbidden = ["我注意到", "根据您提供的信息", "我来提取参数"]
    return not any(w in text for w in forbidden) and len(text) >= 2

def call_llm(client, model, prompt, max_tokens=80):
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
            print(f"LLM调用失败，重试{attempt+1}: {e}")
            time.sleep(2)
    return None

# ======================== 场景生成函数 ========================
def generate_both_empty(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        text_a = format_text_a(None, None)
        prompt = "你是一个正在咨询银行贷款的客户。请生成一个完整的、语义独立的贷款咨询问题。\n要求：问题必须明确、完整，不需要任何上下文也能理解。只输出问题本身。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) > 6:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "COMPLETE"})
    return samples

def generate_only_recent_need(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        recent = random.choice(RECENT_CONVS)
        text_a = format_text_a(None, recent)
        prompt = f"根据下面的最近对话，生成一个用户的追问，这个追问必须依赖对话上下文才能理解（如指代词、省略句、简短追问）。\n\n最近对话：\n{recent}\n\n要求：\n- 生成一个语义不完整的追问（单独看无法理解，必须结合上述对话）\n- 可以是指代词、省略主语的追问、简短的回应\n- 口语化，自然\n- 只输出问题本身\n如果无法生成合适的追问，输出 SKIP。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) <= 15:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "NEED_CONTEXT"})
    return samples

def generate_only_recent_complete(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        recent = random.choice(RECENT_CONVS)
        text_a = format_text_a(None, recent)
        prompt = f"根据下面的最近对话，生成一个用户突然切换话题后的完整提问。\n\n最近对话：\n{recent}\n\n要求：\n- 生成一个语义完整的独立提问\n- 和当前对话话题不同，不依赖上述对话也能理解\n- 口语化、自然\n- 只输出问题本身\n如果无法生成合适的提问，输出 SKIP。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) > 6:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "COMPLETE"})
    return samples

def generate_only_summary_need(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        num = random.randint(1, 2)
        selected = random.sample(SUMMARIES, num)
        summary_text = "\n".join(f"- {s}" for s in selected)
        text_a = format_text_a(summary_text, None)
        prompt = f"根据下面的历史对话摘要，生成一个用户追问。这个追问必须依赖摘要才能理解。\n\n相关对话历史：\n{summary_text}\n\n要求：\n- 生成一个依赖上述摘要的追问（如询问之前提到的具体数值、政策、条件等）\n- 问题本身语义不完整，单独看无法理解具体指代\n- 口语化、自然\n- 只输出问题本身\n如果无法生成合适的追问，输出 SKIP。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) <= 20:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "NEED_CONTEXT"})
    return samples

def generate_only_summary_complete(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        num = random.randint(1, 2)
        selected = random.sample(SUMMARIES, num)
        summary_text = "\n".join(f"- {s}" for s in selected)
        text_a = format_text_a(summary_text, None)
        prompt = f"根据下面的历史对话摘要，生成一个和摘要话题无关的完整贷款咨询问题。\n\n相关对话历史：\n{summary_text}\n\n要求：\n- 生成一个语义完整的独立提问\n- 和上述摘要的话题无关，不依赖摘要也能理解\n- 口语化、自然\n- 只输出问题本身\n如果无法生成合适的提问，输出 SKIP。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) > 6:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "COMPLETE"})
    return samples

def generate_both_need(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        recent = random.choice(RECENT_CONVS)
        num = random.randint(1, 2)
        selected = random.sample(SUMMARIES, num)
        summary_text = "\n".join(f"- {s}" for s in selected)
        text_a = format_text_a(summary_text, recent)
        prompt = f"根据下面的相关对话历史和最近对话，生成一个用户的追问，这个追问必须依赖这些上下文才能理解。\n\n相关对话历史：\n{summary_text}\n\n最近对话：\n{recent}\n\n要求：\n- 生成一个语义不完整的追问（如指代词、省略句、简短追问），必须结合上述上下文才能理解\n- 可以是基于历史摘要的追问，也可以是基于最近对话的追问\n- 口语化，自然\n- 只输出问题本身\n如果无法生成合适的追问，输出 SKIP。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) <= 20:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "NEED_CONTEXT"})
    return samples

def generate_both_complete(client, model, count):
    samples = []
    attempts = 0
    while len(samples) < count and attempts < count * 3:
        attempts += 1
        recent = random.choice(RECENT_CONVS)
        num = random.randint(1, 2)
        selected = random.sample(SUMMARIES, num)
        summary_text = "\n".join(f"- {s}" for s in selected)
        text_a = format_text_a(summary_text, recent)
        prompt = f"根据下面的上下文，生成一个和当前上下文无关的完整贷款咨询问题。\n\n相关对话历史：\n{summary_text}\n\n最近对话：\n{recent}\n\n要求：\n- 生成一个语义完整的独立提问\n- 和上述上下文话题无关，不依赖上下文也能理解\n- 口语化、自然\n- 只输出问题本身\n如果无法生成合适的提问，输出 SKIP。"
        text_b = call_llm(client, model, prompt)
        if text_b and is_valid(text_b) and len(text_b) > 6:
            samples.append({"text_a": text_a, "text_b": text_b, "label": "COMPLETE"})
    return samples

# ======================== 主函数 ========================
def main(args):
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    model = args.model
    random.seed(RANDOM_SEED)

    all_samples = []
    print("生成两者都无场景...")
    all_samples.extend(generate_both_empty(client, model, COUNTS["both_empty"]))
    print("生成只有最近对话场景(NEED_CONTEXT)...")
    all_samples.extend(generate_only_recent_need(client, model, COUNTS["only_recent"]["need"]))
    print("生成只有最近对话场景(COMPLETE)...")
    all_samples.extend(generate_only_recent_complete(client, model, COUNTS["only_recent"]["complete"]))
    print("生成只有摘要场景(NEED_CONTEXT)...")
    all_samples.extend(generate_only_summary_need(client, model, COUNTS["only_summary"]["need"]))
    print("生成只有摘要场景(COMPLETE)...")
    all_samples.extend(generate_only_summary_complete(client, model, COUNTS["only_summary"]["complete"]))
    print("生成两者都有场景(NEED_CONTEXT)...")
    all_samples.extend(generate_both_need(client, model, COUNTS["both"]["need"]))
    print("生成两者都有场景(COMPLETE)...")
    all_samples.extend(generate_both_complete(client, model, COUNTS["both"]["complete"]))

    # 去重
    seen = set()
    unique = []
    for item in all_samples:
        key = (item["text_a"], item["text_b"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"去重后总样本数: {len(unique)}")

    # 分层划分训练集和验证集
    from collections import defaultdict
    by_label = defaultdict(list)
    for item in unique:
        by_label[item["label"]].append(item)

    train, val = [], []
    val_ratio = args.val_ratio
    for label, items in by_label.items():
        random.shuffle(items)
        split = max(1, int(len(items) * (1 - val_ratio)))
        train.extend(items[:split])
        val.extend(items[split:])

    random.shuffle(train)
    random.shuffle(val)

    # 保存训练集
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # 保存验证集
    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n训练集: {OUTPUT_TRAIN} ({len(train)}条)")
    print(f"验证集: {OUTPUT_VAL} ({len(val)}条)")
    for lbl in sorted(by_label.keys()):
        train_cnt = sum(1 for item in train if item["label"] == lbl)
        val_cnt = sum(1 for item in val if item["label"] == lbl)
        print(f"  {lbl}: 训练{train_cnt} 验证{val_cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="验证集比例")
    args = parser.parse_args()
    main(args)