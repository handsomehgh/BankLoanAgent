#!/usr/bin/env python3
"""
Reranker 训练数据生成脚本

根据知识库文档块生成 (query, document, label) 三元组训练样本。
支持三种上下文场景：
    A - 仅对话摘要 (interaction_log)
    B - 对话摘要 + 最近对话 (recent_conv)
    C - 无上下文 (直接独立问题)

正样本：增强查询 + 对应文档块
困难负样本：Embedding 检索排名 10-30 的文档
简单负样本：随机不相关文档

所有上下文仅为生成增强查询使用，最终训练数据不包含上下文。
"""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from modules.module_services.embeddings import RobustLocalEmbeder

# -------------------- 配置 --------------------
MILVUS_URI = "http://192.168.24.128:19530"
COLLECTION_NAME = "business_knowledge"
LLM_API_KEY = "sk-f174be45c6ce4237a4109976bf38c69b"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

OUTPUT_TRAIN = "train_reranker.jsonl"
OUTPUT_VAL = "val_reranker.jsonl"
VAL_RATIO = 0.1
RANDOM_SEED = 42

# 每个文档块生成的增强查询数量
N_QUERY_C_NO_CTX = 1           # 场景 C：无上下文
N_QUERY_A_SUMMARY = 2          # 场景 A：仅有对话摘要
N_QUERY_B_FULL = 3             # 场景 B：对话摘要 + 最近对话

# 每个增强查询配的负样本数量
HARD_NEG_PER_QUERY = 3
EASY_NEG_PER_QUERY = 2

# 困难负样本检索参数
HARD_NEG_RANK_START = 10
HARD_NEG_RANK_END = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 上下文模板 ====================
# 请补充你自己的模板内容，下面为占位示例

INTERACTION_LOG_TEMPLATES = [
    # ==================== 利率相关 (15条) ====================
    "对话摘要：用户刚咨询了房贷利率，助手回复首套房LPR 4.2% + 30BP。",
    "对话摘要：用户询问消费贷利率，助手告知年化利率约3.95%-5.45%。",
    "对话摘要：用户问经营贷利率是不是比房贷低，助手说明了两种贷款的政策差异。",
    "对话摘要：用户问LPR是什么意思，助手解释了贷款市场报价利率的概念。",
    "对话摘要：用户问首套房和二套房利率差多少，助手回复二套房利率不低于LPR+60BP。",
    "对话摘要：用户问公积金贷款利率和商贷差多少，助手回复公积金利率更低。",
    "对话摘要：用户问现在利率是不是历史最低点，助手回复目前利率处于较低水平。",
    "对话摘要：用户问LPR调整后自己的月供会不会变，助手解释了重定价周期。",
    "对话摘要：用户询问固定利率和浮动利率的区别，助手建议根据市场走势选择。",
    "对话摘要：用户想了解助学贷款利率，助手介绍了政策优惠利率。",
    "对话摘要：用户询问经营贷利率优惠政策，助手说明优质客户可享受普惠利率。",
    "对话摘要：用户咨询了车贷利率，助手回复3年期车贷利率约4.5%起。",
    "对话摘要：用户询问装修贷利率，助手告知与消费贷利率相近。",
    "对话摘要：用户问信用卡分期利率和贷款利率哪个高，助手对比了两者。",
    "对话摘要：用户询问利率打折活动，助手说明需关注银行最新优惠。",

    # ==================== 额度相关 (15条) ====================
    "对话摘要：用户询问消费贷额度，助手表示需要评估收入和负债。",
    "对话摘要：用户问最高能贷多少，助手回复需要了解收入和负债情况。",
    "对话摘要：用户提供了月收入2万，助手正在核算可贷额度。",
    "对话摘要：用户问公积金余额能提高贷款额度吗，助手解释了公积金贷款额度计算规则。",
    "对话摘要：用户问夫妻双方收入能否合并计算贷款额度，助手说明可以添加共同借款人。",
    "对话摘要：用户问能否用父母的收入共同借款，助手说明可以添加共同借款人。",
    "对话摘要：用户问月薪5000能贷多少钱，助手回复需要看贷款类型和期限。",
    "对话摘要：用户表示自己收入是现金发放，助手询问是否有其他收入证明。",
    "对话摘要：用户询问抵押经营贷的额度，助手表示需根据抵押物评估值计算。",
    "对话摘要：用户问信用贷和抵押贷额度差异，助手说明抵押贷额度更高。",
    "对话摘要：用户咨询了组合贷款额度分配，助手解释公积金和商贷比例。",
    "对话摘要：用户询问已有房贷再贷消费贷的额度限制，助手说明DTI要求。",
    "对话摘要：用户问企业贷款额度如何核定，助手表示需看经营流水和纳税。",
    "对话摘要：用户询问最高贷款成数，助手说明首套房最高80%。",
    "对话摘要：用户问自己能贷多少，助手引导提供月收入和负债信息。",

    # ==================== 申请条件/资格预审 (15条) ====================
    "对话摘要：用户想了解房贷申请条件，助手列出了基本材料清单。",
    "对话摘要：用户询问年龄限制，助手说明贷款到期时年龄不超过70岁。",
    "对话摘要：用户问工作年限要求，助手回复消费贷需在当前工作满6个月。",
    "对话摘要：用户问征信白户能不能贷款，助手回复可能需要提供更多收入证明。",
    "对话摘要：用户问有逾期记录能否贷款，助手说明需看逾期严重程度。",
    "对话摘要：用户问刚换工作能贷款吗，助手询问试用期是否已过。",
    "对话摘要：用户问个体工商户需要什么条件，助手说明需营业执照满2年。",
    "对话摘要：用户问退休人员能否贷款，助手说明年龄限制会影响贷款期限。",
    "对话摘要：用户问无抵押能贷款吗，助手介绍了信用贷款产品。",
    "对话摘要：用户问外地户口能在本地贷款吗，助手表示需看当地政策。",
    "对话摘要：用户问学生能否贷款，助手说明需要稳定收入来源。",
    "对话摘要：用户问企业贷款需要哪些资质，助手列举了营业执照、纳税证明等。",
    "对话摘要：用户问信用卡逾期记录会影响房贷审批吗，助手表示会参考。",
    "对话摘要：用户问网贷记录多会不会影响银行贷款，助手说明频繁查询有影响。",
    "对话摘要：用户询问担保贷款的条件，助手说明需担保人征信良好。",

    # ==================== 还款方式/月供 (15条) ====================
    "对话摘要：用户咨询了等额本息的计算方式，助手解释了公式。",
    "对话摘要：用户问等额本息和等额本金的区别，助手解释了两种还款方式的概念。",
    "对话摘要：助手建议用户选择等额本息，用户还没有决定。",
    "对话摘要：用户表示等额本金前期压力太大，助手建议可以考虑等额本息。",
    "对话摘要：用户问等额本息前期还的是不是全是利息，助手解释了利息和本金的构成。",
    "对话摘要：用户问月供会不会随着LPR变化，助手解释浮动利率机制。",
    "对话摘要：用户表示对总利息感到惊讶，助手解释等额本息前期利息占比高。",
    "对话摘要：用户问每月还款日是哪天，助手回复一般按放款日确定。",
    "对话摘要：用户问能不能选双周供，助手介绍了双周供的特点。",
    "对话摘要：用户询问先息后本还款方式，助手说明适合短期周转。",
    "对话摘要：用户问提前还款后月供怎么变，助手说明缩短期限或减少月供。",
    "对话摘要：用户咨询了组合贷款的还款方式，助手说明公积金和商贷分开还款。",
    "对话摘要：用户问房贷转按揭后还款方式能否变更，助手说明需重新审批。",
    "对话摘要：用户询问还款日遇到节假日怎么办，助手回复顺延至下一个工作日。",
    "对话摘要：用户问能否只还利息不还本金，助手说明先息后本产品适用。",

    # ==================== 提前还款/展期/逾期 (20条) ====================
    "对话摘要：用户咨询了提前还款政策，助手说明满一年免收违约金。",
    "对话摘要：用户问提前还一部分本金月供会变吗，助手说明可以选择缩短期限或减少月供。",
    "对话摘要：用户问提前还款划算还是继续还贷划算，助手建议进行试算对比。",
    "对话摘要：用户问部分提前还款后剩余本金怎么算，助手解释按剩余本金重新计算。",
    "对话摘要：用户问提前还款预约流程，助手说明需提前1个月申请。",
    "对话摘要：用户询问展期申请条件，助手表示需要评估逾期记录和已还期数。",
    "对话摘要：用户问展期后利率会变吗，助手确认利率将上浮10个基点。",
    "对话摘要：用户问展期后月供能少多少，助手开始试算展期方案。",
    "对话摘要：用户问逾期一天会不会上征信，助手说明一般有宽限期。",
    "对话摘要：用户问还款日忘记还款怎么办，助手建议尽快补还并关注是否产生罚息。",
    "对话摘要：用户问逾期罚息怎么算，助手解释按合同利率1.5倍计算。",
    "对话摘要：用户问逾期记录何时消除，助手说明还清后保留5年。",
    "对话摘要：用户问信用卡逾期和贷款逾期哪个影响大，助手说明贷款逾期更严重。",
    "对话摘要：用户问展期被拒怎么办，助手建议考虑变更还款方式或增加共同还款人。",
    "对话摘要：用户问提前还款违约金能否减免，助手说明特殊情况可申请。",
    "对话摘要：用户问展期后总利息会增加多少，助手进行试算对比。",
    "对话摘要：用户问逾期会影响配偶贷款吗，助手说明夫妻共同借款时会影响。",
    "对话摘要：用户问展期申请需要配偶签字吗，助手说明共同借款需要。",
    "对话摘要：用户问提前还款后征信会变好吗，助手说明正常还款不影响。",
    "对话摘要：用户问展期最长能延多久，助手说明一般不超过原期限一半。",

    # ==================== 材料/流程/政策 (20条) ====================
    "对话摘要：用户问申请房贷需要哪些材料，助手列举了身份证、收入证明、银行流水等。",
    "对话摘要：用户问贷款审批一般需要多久，助手回复信用贷1-3天、房贷1-2周。",
    "对话摘要：用户问面签需要带什么材料，助手说明需要身份证、户口本等原件。",
    "对话摘要：用户问线上能不能申请贷款，助手回复部分产品支持线上申请。",
    "对话摘要：用户问审批通过后多久放款，助手回复抵押登记完成后1-3个工作日。",
    "对话摘要：用户问贷款审批主要看什么，助手说明主要看征信、收入和负债率。",
    "对话摘要：用户问银行流水不够怎么办，助手建议提供其他资产证明。",
    "对话摘要：用户问收入证明怎么开，助手说明需要单位盖章并注明收入金额。",
    "对话摘要：用户问贷款结清后需要办什么手续，助手说明需要办理解除抵押。",
    "对话摘要：用户问贷款合同丢了怎么办，助手回复可以到经办支行补办。",
    "对话摘要：用户问还款卡丢了怎么换卡，助手说明需要到柜台办理还款账户变更。",
    "对话摘要：用户问解押需要哪些材料，助手回复结清证明、身份证、他项权证。",
    "对话摘要：用户问贷后检查会查什么，助手说明主要检查贷款用途和抵押物状况。",
    "对话摘要：用户问续贷需要重新审批吗，助手表示需要重新评估征信和收入。",
    "对话摘要：用户问公积金冲还贷怎么办理，助手讲解月冲和年冲区别。",
    "对话摘要：用户问贷款用途凭证怎么准备，助手说明保留发票和合同。",
    "对话摘要：用户问抵押物被拆迁怎么办，助手说明需提前还款或更换抵押物。",
    "对话摘要：用户问贷款期间能否卖房，助手说明需先还清贷款解除抵押。",
    "对话摘要：用户问房贷转按揭流程，助手说明需评估和重新审批。",
    "对话摘要：用户问贷款保险是否必须买，助手说明房贷通常需要购买。",
]

RECENT_CONV_TEMPLATES = [
    # ==================== 利率/额度 (15条) ====================
    "最近对话：用户: 那房贷利率现在多少？\n助手: 首套房5年以上LPR 4.2%，加30BP后4.5%。",
    "最近对话：用户: 能贷多少？\n助手: 需要了解您的月收入和现有负债情况。",
    "最近对话：用户: 我月入2万，没有负债，最高能贷多少？\n助手: 我帮您算一下最高可贷额度。",
    "最近对话：用户: 消费贷利率能低到多少？\n助手: 目前最低3.95%起，看您的资质。",
    "最近对话：用户: 二套房利率高多少？\n助手: 不低于LPR+60BP，当前约4.95%。",
    "最近对话：用户: 我公积金余额5万，能贷多少？\n助手: 公积金贷款额度受缴存年限和余额影响。",
    "最近对话：用户: 利率是固定的还是浮动的？\n助手: 您可以自行选择，浮动利率随LPR调整。",
    "最近对话：用户: 经营贷利率比房贷低吗？\n助手: 目前经营贷政策性利率更低，但需要有营业执照。",
    "最近对话：用户: 我征信有点花，利率会高吗？\n助手: 征信情况会影响利率定价。",
    "最近对话：用户: 车贷利率现在多少？\n助手: 3年期车贷利率约4.5%起，具体看车型和首付。",
    "最近对话：用户: 夫妻共同贷款额度怎么算？\n助手: 可以合并收入，但负债也会合并计算。",
    "最近对话：用户: 信用贷最高能贷多少？\n助手: 纯信用消费贷一般最高30万。",
    "最近对话：用户: 抵押贷能贷几成？\n助手: 住宅一般可贷评估值的70%-80%。",
    "最近对话：用户: 我月薪5000能贷多少钱？\n助手: 需要看贷款类型和期限，我帮您试算。",
    "最近对话：用户: 你们银行贷款利率有优惠吗？\n助手: 优质客户和首套房可享受优惠利率。",

    # ==================== 还款/月供 (15条) ====================
    "最近对话：用户: 那贷100万30年月供呢？\n助手: 请提供年利率和还款方式。",
    "最近对话：用户: 等额本息和等额本金哪个划算？\n助手: 等额本金总利息更少，但前期压力大。",
    "最近对话：用户: 月供太高了，能降低吗？\n助手: 可以考虑延长贷款期限或变更还款方式。",
    "最近对话：用户: 贷款批下来钱打到哪里？\n助手: 房贷会直接划入开发商或卖方账户。",
    "最近对话：用户: 还款日能不能改？\n助手: 可以申请调整，一般每年可改一次。",
    "最近对话：用户: 月供里包含保险吗？\n助手: 不包含，保险费是单独缴纳的。",
    "最近对话：用户: 先息后本和等额本息哪个好？\n助手: 先息后本适合短期周转，等额本息适合长期贷款。",
    "最近对话：用户: 等额本息每月还款额一样吗？\n助手: 是的，每月还款金额固定，便于预算。",
    "最近对话：用户: 我可以选双周供吗？\n助手: 部分产品支持双周供，可以节省利息。",
    "最近对话：用户: 月供里本金和利息怎么分的？\n助手: 等额本息前期利息占比高，后期本金占比高。",
    "最近对话：用户: 提前还一部分后月供怎么变？\n助手: 可以选择缩短期限或减少月供。",
    "最近对话：用户: 公积金和商贷月供一起扣吗？\n助手: 分开扣款，公积金先扣，余额不足再扣商贷。",
    "最近对话：用户: 还款日遇到周末怎么办？\n助手: 顺延至下一个工作日扣款。",
    "最近对话：用户: 等额本金第几年还清最划算？\n助手: 前期利息高，越早还越省利息。",
    "最近对话：用户: 月供里包括物业费吗？\n助手: 不包括，物业费是单独缴纳的。",

    # ==================== 提前还款/展期/逾期 (15条) ====================
    "最近对话：用户: 提前还贷有违约金吗？\n助手: 满一年通常免收。",
    "最近对话：用户: 提前还20万能省多少利息？\n助手: 请提供剩余本金、年利率和已还期数。",
    "最近对话：用户: 我想一次性还清，需要什么手续？\n助手: 请提供贷款编号，我帮您算一下应还总额。",
    "最近对话：用户: 逾期3天罚息怎么算？\n助手: 请提供逾期本金和合同年利率。",
    "最近对话：用户: 我忘了还款，现在补上还来得及吗？\n助手: 一般在3天宽限期内不算逾期。",
    "最近对话：用户: 我能申请展期吗？\n助手: 请问已还了多少期？有无逾期记录？",
    "最近对话：用户: 展期后月供能少多少？\n助手: 我帮您试算一下，请补充当前剩余本金和利率。",
    "最近对话：用户: 缩短期限和减少月供哪个划算？\n助手: 缩短期限节省利息更多，减少月供减轻压力。",
    "最近对话：用户: 逾期会影响征信吗？\n助手: 一旦上报征信就会留下记录，建议尽快还款。",
    "最近对话：用户: 展期后利率会变吗？\n助手: 一般会适度上浮，我帮您具体算一下。",
    "最近对话：用户: 提前还款预约需要多久？\n助手: 一般需要提前1个月向经办行申请。",
    "最近对话：用户: 我有过逾期还能展期吗？\n助手: 需要看逾期次数和当前状态。",
    "最近对话：用户: 罚息比正常利息高多少？\n助手: 一般按合同利率的1.5倍计算。",
    "最近对话：用户: 提前还款后征信会变好吗？\n助手: 正常提前还款对征信无负面影响。",
    "最近对话：用户: 逾期记录什么时候能消除？\n助手: 从还清那天算起，5年后自动删除。",

    # ==================== 材料/流程/申请 (20条) ====================
    "最近对话：用户: 我需要准备哪些材料？\n助手: 身份证、收入证明、银行流水。",
    "最近对话：用户: 贷款审批要多久？\n助手: 信用贷1-3天，房贷约1-2周。",
    "最近对话：用户: 面签需要带什么？\n助手: 身份证、户口本、结婚证等原件。",
    "最近对话：用户: 银行流水不够怎么办？\n助手: 可以提供其他资产证明，如房产、存单。",
    "最近对话：用户: 收入证明怎么开？\n助手: 需要单位盖章并注明月收入金额。",
    "最近对话：用户: 审批通过后多久放款？\n助手: 抵押登记完成后1-3个工作日。",
    "最近对话：用户: 贷款合同丢了怎么办？\n助手: 可以到经办支行申请复印件。",
    "最近对话：用户: 解押需要哪些材料？\n助手: 结清证明、身份证、他项权证。",
    "最近对话：用户: 还款卡丢了怎么变更？\n助手: 带新卡和身份证到柜台办理。",
    "最近对话：用户: 公积金冲还贷怎么办理？\n助手: 到公积金中心或手机APP签约办理。",
    "最近对话：用户: 贷后检查会查什么？\n助手: 主要检查贷款用途是否合规、抵押物状况。",
    "最近对话：用户: 续贷需要重新审批吗？\n助手: 需要重新评估您的征信和收入情况。",
    "最近对话：用户: 贷款期间能卖房吗？\n助手: 需要先还清贷款解除抵押才能过户。",
    "最近对话：用户: 抵押物被拆迁了怎么办？\n助手: 需要用拆迁款提前还款或更换抵押物。",
    "最近对话：用户: 贷款用途凭证怎么准备？\n助手: 保留好发票和合同，按要求上传。",
    "最近对话：用户: 房贷转按揭怎么操作？\n助手: 需要向新银行申请，重新评估审批。",
    "最近对话：用户: 贷款保险必须买吗？\n助手: 房贷通常需要购买抵押物财产保险。",
    "最近对话：用户: 线上能申请贷款吗？\n助手: 纯信用消费贷可全程手机银行办理。",
    "最近对话：用户: 审批没通过还能再申请吗？\n助手: 建议改善征信和负债后3个月再试。",
    "最近对话：用户: 贷款下来后能直接取现吗？\n助手: 消费贷会打入您账户，可以取用。",

    # ==================== 产品对比/其他 (15条) ====================
    "最近对话：用户: 房贷和消费贷有什么区别？\n助手: 房贷用于买房，利率低、期限长；消费贷用于日常消费，利率高、期限短。",
    "最近对话：用户: 抵押贷和信用贷哪个好？\n助手: 抵押贷利率低额度高，信用贷手续简便放款快。",
    "最近对话：用户: 公积金贷款和商贷哪个划算？\n助手: 公积金利率更低，但额度有限，组合贷款是不错的选择。",
    "最近对话：用户: 经营贷需要什么条件？\n助手: 营业执照满2年、经营流水、纳税证明。",
    "最近对话：用户: 装修贷和消费贷哪个利率低？\n助手: 利率相近，装修贷有特定用途限制。",
    "最近对话：用户: 助学贷款怎么申请？\n助手: 需提供录取通知书和家庭经济困难证明。",
    "最近对话：用户: 车位贷能贷多少？\n助手: 一般不超过车位评估值的70%。",
    "最近对话：用户: 信用卡分期和贷款哪个划算？\n助手: 贷款通常利率更低，分期手续费较高。",
    "最近对话：用户: 网贷和银行贷款有什么区别？\n助手: 银行利率更低、更安全，网贷门槛低但利率高。",
    "最近对话：用户: 组合贷款公积金和商贷比例怎么定？\n助手: 公积金部分按额度上限，剩余用商贷补齐。",
    "最近对话：用户: 我能用父母的公积金贷款吗？\n助手: 父母可以作为共同借款人使用公积金。",
    "最近对话：用户: 贷款下来后还能追加额度吗？\n助手: 需要重新审批，不能直接追加。",
    "最近对话：用户: 房贷批了还能改还款方式吗？\n助手: 放款前可以修改，放款后需要申请变更。",
    "最近对话：用户: 贷款期间利率会变吗？\n助手: 浮动利率会随LPR调整，固定利率不变。",
    "最近对话：用户: 征信不好能贷款吗？\n助手: 可以尝试申请，但可能影响额度和利率。",
]




class RerankerDataGenerator:
    def __init__(self, milvus_uri: str, collection: str, api_key: str, base_url: str, model: str):
        # 连接 Milvus
        connections.connect(alias="default", uri=milvus_uri)
        self.collection = Collection(collection)
        self.collection.load()
        self.chunks = self._load_chunks()
        self.chunk_texts = [c["text"] for c in self.chunks]

        # 加载 Embedding 模型
        self.embedder = RobustLocalEmbeder(base_url="http://localhost:8000/v1",model_name="",dimensions=512)
        self.chunk_embeddings = self.embedder.embed_documents(self.chunk_texts)

        # LLM 客户端
        self.llm = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = model

    def _load_chunks(self) -> List[Dict]:
        logger.info("正在从 Milvus 加载知识库文档块...")
        results = self.collection.query(
            expr="id != ''",
            output_fields=["id", "text", "source_type", "product_type"],
            limit=5000
        )
        logger.info(f"加载了 {len(results)} 个文档块")
        return results

    def _generate_enhanced_query(
        self,
        chunk_text: str,
        interaction_log: Optional[str] = None,
        recent_conv: Optional[str] = None
    ) -> Optional[str]:
        """
        调用 LLM 生成一个增强查询，根据传入的上下文类型自动适配 prompt。
        - 无上下文：场景 C
        - 仅有 interaction_log：场景 A
        - 两者都有：场景 B
        """
        if not interaction_log and not recent_conv:
            # 场景 C：无上下文
            prompt = f"""你是一个正在咨询银行贷款的用户。请生成一个完整的、自然的用户提问，这个提问的答案应该能够从下面的知识库文档中找到：
{chunk_text[:800]}

要求：
- 提问必须是一个完整的、独立的句子，不依赖任何上下文。
- 口语化、自然。
- 只输出提问本身，不要任何解释。"""
        elif interaction_log and not recent_conv:
            # 场景 A：仅有对话摘要
            prompt = f"""你是一个正在咨询银行贷款的用户。以下是之前对话的摘要：
{interaction_log}

现在，请根据这个对话摘要，生成一个完整的、自然的用户提问，这个提问的答案应该能够从下面的知识库文档中找到：
{chunk_text[:800]}

要求：
- 提问应延续之前的话题，但问的是一个新的、更具体的方面。
- 必须是完整的句子，不能使用省略或指代（如“那30年呢”）。
- 口语化、自然。
- 只输出提问本身，不要任何解释。"""
        else:
            # 场景 B：对话摘要 + 最近对话
            prompt = f"""你是一个正在咨询银行贷款的用户。以下是之前的对话摘要和最近的对话记录：

【对话摘要】
{interaction_log}

【最近对话】
{recent_conv}

现在，请根据这些信息，生成一个完整的、自然的用户提问，这个提问的答案应该能够从下面的知识库文档中找到：
{chunk_text[:800]}

要求：
- 提问应体现你正在回应助手的最新消息（可能是追问细节、补充参数或提出新问题）。
- 必须是完整的句子，不能使用省略或指代（如“那30年呢”）。
- 如果对话摘要和最近对话存在话题不一致或冲突，请以最近对话为准进行提问。
- 如果最近对话与知识库文档完全不相关，可以直接生成一个基于最近对话的完整提问，忽略文档内容。
- 口语化、自然。
- 只输出提问本身，不要任何解释。"""

        try:
            resp = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM 生成查询失败: {e}")
            return None

    def _retrieve_hard_negatives(self, query: str, positive_idx: int, count: int) -> List[str]:
        """检索困难负样本：排名在 [start, end] 的文档"""
        query_vec = self.embedder.embed_documents([query])[0]
        scores = np.dot(self.chunk_embeddings, query_vec)
        scores[positive_idx] = -float("inf")
        top_indices = np.argsort(scores)[::-1][:HARD_NEG_RANK_END]
        candidates = top_indices[HARD_NEG_RANK_START:]
        if len(candidates) >= count:
            selected = random.sample(list(candidates), count)
        else:
            selected = candidates[:count]
        return [self.chunk_texts[i] for i in selected]

    def _random_easy_negatives(self, exclude_indices: set, count: int) -> List[str]:
        """随机采样简单负样本（排除正样本和困难负样本）"""
        pool = [i for i in range(len(self.chunk_texts)) if i not in exclude_indices]
        selected = random.sample(pool, min(count, len(pool)))
        return [self.chunk_texts[i] for i in selected]

    def generate_samples(self) -> List[Dict]:
        all_samples = []
        for idx, chunk in enumerate(self.chunks):
            logger.info(f"处理文档块 {idx+1}/{len(self.chunks)} (ID: {chunk['id']})")
            chunk_text = chunk["text"]

            # -------- 场景 C：无上下文 --------
            for _ in range(N_QUERY_C_NO_CTX):
                q = self._generate_enhanced_query(chunk_text)
                if not q:
                    continue
                all_samples.append({"query": q, "document": chunk_text, "label": 1})
                hard_negs = self._retrieve_hard_negatives(q, idx, HARD_NEG_PER_QUERY)
                for neg in hard_negs:
                    all_samples.append({"query": q, "document": neg, "label": 0})
                exclude = {idx} | {self.chunk_texts.index(n) for n in hard_negs if n in self.chunk_texts}
                easy_negs = self._random_easy_negatives(exclude, EASY_NEG_PER_QUERY)
                for neg in easy_negs:
                    all_samples.append({"query": q, "document": neg, "label": 0})

            # -------- 场景 A：仅有对话摘要 --------
            for _ in range(N_QUERY_A_SUMMARY):
                summary = random.choice(INTERACTION_LOG_TEMPLATES)
                q = self._generate_enhanced_query(chunk_text, interaction_log=summary)
                if not q:
                    continue
                all_samples.append({"query": q, "document": chunk_text, "label": 1})
                hard_negs = self._retrieve_hard_negatives(q, idx, HARD_NEG_PER_QUERY)
                for neg in hard_negs:
                    all_samples.append({"query": q, "document": neg, "label": 0})
                exclude = {idx} | {self.chunk_texts.index(n) for n in hard_negs if n in self.chunk_texts}
                easy_negs = self._random_easy_negatives(exclude, EASY_NEG_PER_QUERY)
                for neg in easy_negs:
                    all_samples.append({"query": q, "document": neg, "label": 0})

            # -------- 场景 B：对话摘要 + 最近对话（随机组合）--------
            for _ in range(N_QUERY_B_FULL):
                summary = random.choice(INTERACTION_LOG_TEMPLATES)
                recent = random.choice(RECENT_CONV_TEMPLATES)
                q = self._generate_enhanced_query(chunk_text, interaction_log=summary, recent_conv=recent)
                if not q:
                    continue
                all_samples.append({"query": q, "document": chunk_text, "label": 1})
                hard_negs = self._retrieve_hard_negatives(q, idx, HARD_NEG_PER_QUERY)
                for neg in hard_negs:
                    all_samples.append({"query": q, "document": neg, "label": 0})
                exclude = {idx} | {self.chunk_texts.index(n) for n in hard_negs if n in self.chunk_texts}
                easy_negs = self._random_easy_negatives(exclude, EASY_NEG_PER_QUERY)
                for neg in easy_negs:
                    all_samples.append({"query": q, "document": neg, "label": 0})

        return all_samples

    def save(self, samples: List[Dict], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"保存 {len(samples)} 条样本至 {output_path}")

    @staticmethod
    def split_train_val(samples: List[Dict], val_ratio: float = VAL_RATIO
                        ) -> Tuple[List[Dict], List[Dict]]:
        """分层切分训练/验证集，保持正负样本比例"""
        pos_samples = [s for s in samples if s["label"] == 1]
        neg_samples = [s for s in samples if s["label"] == 0]

        random.shuffle(pos_samples)
        random.shuffle(neg_samples)

        pos_split = int(len(pos_samples) * (1 - val_ratio))
        neg_split = int(len(neg_samples) * (1 - val_ratio))

        train_samples = pos_samples[:pos_split] + neg_samples[:neg_split]
        val_samples = pos_samples[pos_split:] + neg_samples[neg_split:]

        random.shuffle(train_samples)
        random.shuffle(val_samples)

        logger.info(f"分层切分完毕：训练集 {len(train_samples)} 条，验证集 {len(val_samples)} 条")
        logger.info(f"  训练集正样本: {pos_split}，负样本: {neg_split}")
        logger.info(f"  验证集正样本: {len(pos_samples) - pos_split}，负样本: {len(neg_samples) - neg_split}")
        return train_samples, val_samples

    def save_samples(self, samples: List[Dict], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"保存 {len(samples)} 条样本至 {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=LLM_API_KEY)
    parser.add_argument("--base_url", type=str, default=LLM_BASE_URL)
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO, help="验证集比例 (默认 0.1)")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    gen = RerankerDataGenerator(
        milvus_uri=MILVUS_URI,
        collection=COLLECTION_NAME,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )
    all_samples = gen.generate_samples()
    logger.info(f"生成总样本: {len(all_samples)} 条")

    # 分层切分
    train_samples, val_samples = RerankerDataGenerator.split_train_val(all_samples, args.val_ratio)

    # 保存
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen.save_samples(train_samples, str(out_dir / OUTPUT_TRAIN))
    gen.save_samples(val_samples, str(out_dir / OUTPUT_VAL))
    logger.info(f"数据已保存到 {out_dir}")


if __name__ == "__main__":
    main()