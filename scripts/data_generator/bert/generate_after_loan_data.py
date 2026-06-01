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
OUTPUT_TRAIN = "after_train.jsonl"
OUTPUT_VAL = "after_val.jsonl"
RANDOM_SEED = 42

# ======================== 合法标签 ========================
ALL_LABELS = [
    "DIRECT_REPLY",
    "CLARIFY",
    "prepayment_evaluation_skill",
    "extension_management_skill",
    "overdue_handling_skill",
    "repayment_method_switch_skill",
    "calculate_prepayment",
    "check_extension_eligibility",
    "calculate_extension_plan",
    "calculate_overdue_penalty",
    "calculate_repayment_method_switch",
    "generate_repayment_schedule",
    "generate_settlement_certificate",
    "general_search_knowledge",
]

# ======================== 扩展的上下文模板库 ========================
# (此处沿用你之前确认过的完整模板库，保持不变)
CONTEXT_TEMPLATES = {
    "profile_only": [
        # 原有 15 条
        "用户画像：月收入1.5万，现有房贷100万，期限30年，已还3年，等额本息。",
        "用户画像：月收入2万，有消费贷20万，已还1年，等额本金。",
        "用户画像：个体户，年收入30万，经营贷80万，期限5年，已还2年。",
        "用户画像：退休人员，月退休金8000元，房贷已还清，欲申请装修贷。",
        "用户画像：自由职业，收入不稳定，房贷50万，已还5年，有3次逾期记录。",
        "用户画像：国企员工，月入1.8万，公积金贷款60万30年，已还8年，等额本息。",
        "用户画像：私企职员，月入1.2万，消费贷10万3年，已还1年半，无逾期。",
        "用户画像：小微企业主，经营贷150万10年，已还4年，等额本金，有1次逾期已还清。",
        "用户画像：教师，月入1万，组合贷款80万20年，已还6年，等额本息。",
        "用户画像：快递员，月入9000元，车贷8万3年，已还1年，等额本息，征信良好。",
        "用户画像：医生，月入2.5万，房贷120万25年，已还2年，想提前还款。",
        "用户画像：程序员，月入3万，房贷200万30年，已还1年，等额本息，无负债。",
        "用户画像：宝妈，家庭月入3万（配偶），名下消费贷5万，已还半年。",
        "用户画像：刚退休，公积金贷款已还清，想办理解除抵押手续。",
        "用户画像：出租车司机，经营贷30万5年，已还3年，有2次逾期记录。",
        # 新增 15 条
        "用户画像：会计，月入1.4万，房贷65万15年，等额本金，已还4年，想提前还清。",
        "用户画像：销售经理，月入2.2万（含提成），车贷15万4年，已还1.5年，等额本息。",
        "用户画像：护士，月入1.1万，消费贷8万2年，已还10个月，无逾期，想变更还款方式。",
        "用户画像：律师，年收入50万，经营贷200万10年，已还3年，等额本息，考虑展期。",
        "用户画像：外卖骑手，月收入不稳定约1.2万，车贷6万2年，已还9个月，逾期1次已还。",
        "用户画像：公务员，月入1.6万，公积金贷款90万30年，已还12年，等额本息。",
        "用户画像：企业高管，月薪5万，房贷300万25年，已还2年，等额本息，想一次性还清。",
        "用户画像：大学生，毕业后刚工作半年，月入8000，消费贷3万1年，已还3个月。",
        "用户画像：餐厅老板，经营贷50万5年，已还3年，受疫情影响收入下降，想申请展期。",
        "用户画像：建筑工人，月入1.3万（现金），房贷40万20年，已还8年，等额本息，征信空白。",
        "用户画像：基金经理，月入8万，抵押经营贷500万10年，已还1年，等额本金。",
        "用户画像：自媒体人，收入波动大，平均月入2万，消费贷12万3年，已还1年。",
        "用户画像：退休教师，月退休金1.2万，名下无贷款，想为子女购房做担保咨询贷后政策。",
        "用户画像：海员，年收入20万（集中发放），房贷70万20年，已还3年，等额本息，有6个月未还记录但已补齐。",
        "用户画像：工厂技工，月入1万，车贷10万3年，已还2年，等额本金，想提前还清拿回车辆登记证。",
    ],
    "summary_only": [
        # 原有 15 条
        "对话摘要：用户刚咨询了提前还款政策，助手说明满一年免收违约金。",
        "对话摘要：用户询问展期申请条件，助手表示需要评估逾期记录和已还期数。",
        "对话摘要：用户想了解逾期罚息计算方式，助手解释了日利率和罚息倍数。",
        "对话摘要：用户表达了对月供过高的不满，助手建议考虑展期或变更还款方式。",
        "对话摘要：用户询问结清证明开具流程，助手告知需携带身份证和贷款合同到网点办理。",
        "对话摘要：用户想了解提前还款后月供变化，助手表示需要根据剩余本金重新计算。",
        "对话摘要：用户咨询房贷解押流程，助手说明了需要准备的材料和办理地点。",
        "对话摘要：用户询问续贷条件，助手表示需评估当前征信和负债情况。",
        "对话摘要：用户对还款日扣款失败表示担忧，助手解释宽限期政策。",
        "对话摘要：用户想比较缩短期限和减少月供两种提前还款方案，助手开始试算。",
        "对话摘要：用户询问展期后利率是否会变，助手确认利率将上浮10个基点。",
        "对话摘要：用户想了解如何查询剩余还款期数，助手告知可通过手机银行查看。",
        "对话摘要：用户表示想一次性结清贷款，助手说明需要提前预约并计算应还总额。",
        "对话摘要：用户询问能否将等额本息改为等额本金，助手表示可以申请变更。",
        "对话摘要：用户想了解逾期记录何时消除，助手解释自还清之日起保留5年。",
        # 新增 15 条
        "对话摘要：用户咨询了提前还款预约流程，助手告知需提前1个月向经办行申请。",
        "对话摘要：用户询问部分提前还款后是否可以选择缩短期限，助手确认可以并开始试算。",
        "对话摘要：用户想知道展期申请被拒后还能怎么办，助手建议考虑变更还款方式或增加共同还款人。",
        "对话摘要：用户对房贷利率下调后自己的月供是否变化有疑问，助手解释浮动利率重定价规则。",
        "对话摘要：用户咨询了贷后检查被抽中怎么办，助手说明需配合提供资金用途凭证。",
        "对话摘要：用户想了解抵押物被拆迁后贷款如何处理，助手解释需提前还款或更换抵押物。",
        "对话摘要：用户询问贷款期间能否出售抵押房产，助手说明需先还清贷款解除抵押。",
        "对话摘要：用户表示配偶去世，咨询贷款继承和还款责任问题。",
        "对话摘要：用户想了解因疫情导致的逾期能否申请征信修复，助手说明可提供证明材料申请。",
        "对话摘要：用户询问公积金冲还贷的具体操作，助手讲解了月冲和年冲的区别。",
        "对话摘要：用户想了解提前还款是否会影响个人征信，助手说明正常提前还款对征信无负面影响。",
        "对话摘要：用户咨询贷款合同遗失后如何补办，助手告知可到经办行申请复印件。",
        "对话摘要：用户询问还款卡丢失后如何变更还款账户，助手说明需本人携带新卡和身份证到柜台办理。",
        "对话摘要：用户想了解贷款期间能否增加共同借款人，助手说明需要重新审批。",
        "对话摘要：用户咨询了房贷转按揭到其他银行的流程和费用。",
    ],
    "recent_conv_only": [
        # 提前还款相关
        "最近对话：用户: 提前还20万能省多少利息？\n助手: 请提供剩余本金、年利率和已还期数。",
        "最近对话：用户: 我想一次性还清，需要什么手续？\n助手: 请提供贷款编号，我帮您算一下应还总额。",
        "最近对话：用户: 提前还款违约金怎么收？\n助手: 满一年免收，不满一年按剩余本金的1%收取。",
        "最近对话：用户: 缩短期限和减少月供哪个划算？\n助手: 我帮您对比一下两种方案。",
        "最近对话：用户: 还了5年了，现在提前还可免违约金吧？\n助手: 是的，满一年即可免收。",
        # 展期相关
        "最近对话：用户: 我能申请展期吗？\n助手: 请问已还了多少期？有无逾期记录？",
        "最近对话：用户: 展期后月供能少多少？\n助手: 我帮您试算一下，请补充当前剩余本金和利率。",
        "最近对话：用户: 展期后利率会变吗？\n助手: 一般会适度上浮，我帮您具体算一下。",
        "最近对话：用户: 我有过逾期还能展期吗？\n助手: 需要看逾期次数和当前状态。",
        "最近对话：用户: 最多能展期多长时间？\n助手: 一般不超过原期限的一半。",
        # 逾期相关
        "最近对话：用户: 逾期3天罚息怎么算？\n助手: 请提供逾期本金和合同年利率。",
        "最近对话：用户: 我忘了还款，现在补上还来得及吗？\n助手: 一般在3天宽限期内不算逾期。",
        "最近对话：用户: 逾期会影响征信吗？\n助手: 一旦上报征信就会留下记录，建议尽快还款。",
        "最近对话：用户: 罚息比正常利息高多少？\n助手: 一般按合同利率的1.5倍计算。",
        # 还款方式变更
        "最近对话：用户: 等额本息改成等额本金划算吗？\n助手: 我需要知道贷款金额、利率和已还期数才能试算。",
        "最近对话：用户: 改还款方式要手续费吗？\n助手: 一般收取200元变更手续费。",
        "最近对话：用户: 改成等额本金后月供会不会太高？\n助手: 前期月供会比现在高，我帮您算一下具体数字。",
        # 还款计划/结清证明
        "最近对话：用户: 我要一份还款计划表。\n助手: 请提供贷款本金、年利率和期限。",
        "最近对话：用户: 贷款还清了，我要开结清证明。\n助手: 请提供姓名和贷款编号。",
        "最近对话：用户: 结清证明需要盖章吗？\n助手: 需要银行公章，然后到不动产登记中心办理解押。",
        # 政策/流程咨询
        "最近对话：用户: 解押需要哪些材料？\n助手: 结清证明、身份证、他项权证。",
        "最近对话：用户: 贷后检查会查什么？\n助手: 主要检查贷款用途是否合规、抵押物状况。",
        "最近对话：用户: 续贷需要重新审批吗？\n助手: 需要重新评估您的征信和收入情况。",
        "最近对话：用户: 还款日能不能改？\n助手: 可以申请调整，一般每年可改一次。",
        # 情绪化表达
        "最近对话：用户: 月供压力太大了，快还不上了。\n助手: 我理解您的压力，我们可以看看展期或变更还款方式的可能性。",
        "最近对话：用户: 逾期一天就上征信，太不合理了！\n助手: 我理解您的不满，一般有宽限期，具体看合同约定。",
        "最近对话：用户: 我房贷还剩80万，想提前还30万，剩下50万怎么算？\n助手: 需要确认缩短期限还是减少月供，两种方案我都帮您算。",
        "最近对话：用户: 经营贷提前还款要预约吗？\n助手: 需要提前1个月向经办行申请，并填写提前还款申请表。",
        "最近对话：用户: 我刚发年终奖，想一次性还清车贷。\n助手: 我帮您算一下剩余本金和可能的违约金。",
        "最近对话：用户: 提前还房贷会影响我的征信吗？\n助手: 正常提前还款不会对征信产生负面影响。",
        # 新增展期相关
        "最近对话：用户: 我失业了，房贷还不上了，能延期吗？\n助手: 我理解您的处境，我先帮您查一下展期资格。",
        "最近对话：用户: 展期后我的贷款利率会变吗？\n助手: 一般会适度上浮，具体幅度取决于您的贷款类型和征信情况。",
        "最近对话：用户: 展期申请需要配偶签字吗？\n助手: 如果贷款时是夫妻共同借款，需要双方同意。",
        # 新增逾期相关
        "最近对话：用户: 我逾期5天了，会不会已经上征信了？\n助手: 一般银行在逾期30天以上才上报征信，但建议尽快还款。",
        "最近对话：用户: 我之前逾期过，现在想提前还款，银行会为难我吗？\n助手: 只要当前没有逾期，提前还款是您的权利。",
        "最近对话：用户: 信用卡逾期和贷款逾期哪个影响大？\n助手: 都影响征信，但贷款逾期对后续贷款审批影响更大。",
        # 新增还款方式变更
        "最近对话：用户: 我收入越来越高了，等额本金前期压力大但总利息少，我该不该换？\n助手: 如果当前现金流充裕，改成等额本金能省不少利息。",
        "最近对话：用户: 我已经还了10年了，现在改成等额本金还有意义吗？\n助手: 剩余期限越长，变更节省的利息越多，我帮您算一下。",
        # 新增结清证明/解押
        "最近对话：用户: 我的贷款还清了，但是房子还在抵押状态，怎么解押？\n助手: 您需要先开具结清证明，然后携带相关材料到不动产登记中心办理解押。",
        "最近对话：用户: 结清证明可以代办吗？\n助手: 一般需要本人办理，特殊情况可公证委托他人。",
        # 新增其他贷后场景
        "最近对话：用户: 我贷款买的房子要拆迁了，贷款怎么办？\n助手: 您可以用拆迁款提前还清贷款，或者与银行协商更换抵押物。",
        "最近对话：用户: 我想把房子卖掉，但贷款还没还清，怎么操作？\n助手: 需要先还清贷款解除抵押，或者买家同意承接贷款（转按揭）。",
        "最近对话：用户: 我父亲去世了，他的房贷我要继续还吗？\n助手: 如果您继承了房产，相应贷款也需要继承，或者选择放弃继承。",
        "最近对话：用户: 我的还款卡丢了，换了新卡号，怎么变更？\n助手: 请携带新卡和身份证到经办行柜台办理还款账户变更。",
        "最近对话：用户: 贷款合同找不到了，还能办业务吗？\n助手: 可以到经办行申请复印件，带身份证即可。",
        "最近对话：用户: 公积金按月冲还贷怎么办理？\n助手: 需要到公积金中心或通过手机公积金APP签约办理。",
        "最近对话：用户: 我想增加我老婆为共同还款人，需要什么手续？\n助手: 需要夫妻双方携带身份证、结婚证到银行重新审批。",
        "最近对话：用户: 提前还款后，我的月供不变，期限缩短了，这样划算吗？\n助手: 这种方案总利息节省最多，是银行推荐的默认方案。",
        "最近对话：用户: 贷款期间我能把房子过户给儿子吗？\n助手: 需要先还清贷款解除抵押才能办理过户。",
        "最近对话：用户: 我的经营贷贷后检查需要提供哪些材料？\n助手: 一般需要经营流水、纳税证明、贷款用途凭证等。",
        "最近对话：用户: 续贷和展期有什么区别？\n助手: 续贷是重新申请一笔新贷款，展期是延长原有贷款期限。",
    ],
    "tool_ops_only": [
        "工具操作：助手: calculate_prepayment\n工具结果: 缩短期限可节省利息13.2万，违约金0元。",
        "工具操作：助手: calculate_prepayment\n工具结果: 减少月供方案：新月供4320元，节省利息8.5万。",
        "工具操作：助手: check_extension_eligibility\n工具结果: 符合展期条件，需携带身份证和收入证明。",
        "工具操作：助手: check_extension_eligibility\n工具结果: 因逾期记录，暂不符合展期条件。",
        "工具操作：助手: calculate_extension_plan\n工具结果: 展期后月供从5800降至4100元，总利息增加3.2万。",
        "工具操作：助手: calculate_overdue_penalty\n工具结果: 逾期罚息128.5元，应还总额210,128.5元。",
        "工具操作：助手: calculate_repayment_method_switch\n工具结果: 变更后节省利息约5.2万，手续费200元。",
        "工具操作：助手: generate_repayment_schedule\n工具结果: 已生成36期还款计划表，首月月供6125元。",
        "工具操作：助手: generate_settlement_certificate\n工具结果: 结清证明模板已生成，需加盖银行公章。",
        "工具操作：助手: general_search_knowledge\n工具结果: 解押流程需携带结清证明、身份证、他项权证到不动产登记中心办理。",
        "工具操作：助手: prepayment_evaluation_skill\n工具结果: 综合评估完成：缩短期限方案净节省12.8万，推荐此方案。",
        "工具操作：助手: extension_management_skill\n工具结果: 展期资格通过，展期36个月方案月供降幅最大。",
        "工具操作：助手: overdue_handling_skill\n工具结果: 罚息已计算，建议在宽限期内尽快还款避免征信影响。",
        "工具操作：助手: repayment_method_switch_skill\n工具结果: 变更为等额本金后总利息减少4.8万，首月月供增加至6500元。",
        "工具操作：助手: calculate_prepayment\n工具结果: 全部结清方案：应还总额85.2万，其中违约金3200元（不满一年）。",
        "工具操作：助手: check_extension_eligibility\n工具结果: 展期资格通过，但需补充配偶收入证明。",
        "工具操作：助手: calculate_extension_plan\n工具结果: 展期24个月，月供降低1500元，总利息增加2.8万。",
        "工具操作：助手: calculate_overdue_penalty\n工具结果: 逾期10天，罚息456.2元，已产生征信记录。",
        "工具操作：助手: generate_repayment_schedule\n工具结果: 已生成240期还款计划，已还60期，剩余180期。",
        "工具操作：助手: general_search_knowledge\n工具结果: 夫妻共同借款人变更需重新签订借款合同。",
        "工具操作：助手: prepayment_evaluation_skill\n工具结果: 缩短期限方案净节省15.6万，减少月供方案净节省9.3万。",
        "工具操作：助手: extension_management_skill\n工具结果: 因当前有未结清逾期，展期申请被拒。",
        "工具操作：助手: overdue_handling_skill\n工具结果: 罚息已生成，建议立即还款，逾期记录将在还清后保留5年。",
        "工具操作：助手: repayment_method_switch_skill\n工具结果: 变更后总利息节省6.8万，但首月月供增加至7200元。",
    ],
    "mixed_rich": [
        "用户画像：月入2万，房贷80万20年，等额本息，已还5年。\n对话摘要：用户想提前还部分本金并变更还款方式。\n最近对话：用户: 提前还30万，改成等额本金，能省多少？\n助手: 我帮您综合评估一下。",
        "用户画像：月入1万，消费贷15万3年，等额本金，已还1年，有2次逾期。\n对话摘要：用户担心逾期影响，咨询展期事宜。\n工具操作：助手: check_extension_eligibility\n工具结果: 因逾期记录，暂不符合展期条件。",
        "用户画像：退休，月退休金8000元，房贷已还清。\n对话摘要：用户想办结清证明。\n最近对话：用户: 贷款还完了，怎么开证明？\n助手: 我帮您生成模板，然后到网点盖章。",
        "用户画像：个体户，经营贷100万10年，已还6年，等额本金。\n对话摘要：用户想评估提前还款是否划算。\n工具操作：助手: calculate_prepayment\n工具结果: 全部结清可节省利息25.8万，违约金0元。\n最近对话：用户: 那帮我约一下提前还款。",
        "用户画像：国企员工，公积金贷款70万30年，已还10年，等额本息。\n对话摘要：用户想查询剩余还款计划。\n最近对话：用户: 帮我看一下还有多少期，每期还多少。\n助手: 我帮您生成完整的还款计划表。",
        "用户画像：宝妈，家庭月入3万，消费贷已还清。\n对话摘要：用户想了解解押流程。\n工具操作：助手: general_search_knowledge\n工具结果: 解押需要结清证明、身份证、他项权证。\n最近对话：用户: 这些材料在哪里办？",
        "用户画像：快递员，车贷8万3年，已还2年，等额本息。\n对话摘要：用户逾期2天担心上征信。\n最近对话：用户: 我忘了还款，逾期2天了怎么办？\n助手: 先别急，一般在3天宽限期内补还不会上征信。",
        "用户画像：教师，组合贷款80万20年，已还8年，等额本息。\n对话摘要：用户想变更还款方式。\n工具操作：助手: calculate_repayment_method_switch\n工具结果: 变更后总利息节省6.1万，手续费200元。\n最近对话：用户: 那手续费怎么交？",
        "用户画像：程序员，房贷200万30年，已还1年，等额本息。\n对话摘要：用户觉得月供太高想提前还一部分。\n最近对话：用户: 我年终奖发了，想提前还50万。\n助手: 我帮您算一下缩短期限和减少月供两个方案。",
        "用户画像：医生，房贷120万25年，已还3年，等额本金。\n对话摘要：用户想了解提前还款后能否变更还款方式。\n最近对话：用户: 提前还一部分后，剩下的能改成等额本息吗？\n助手: 可以，我帮您综合试算。",
        "用户画像：销售经理，月入2.2万，车贷15万4年，已还1.5年。\n对话摘要：用户想一次性还清车贷拿回车辆登记证。\n最近对话：用户: 我还剩多少钱没还？一次性还清要违约金吗？\n助手: 我帮您算一下剩余本金和可能的违约金。",
        "用户画像：会计，房贷65万15年，等额本金，已还4年。\n对话摘要：用户想提前还清。\n工具操作：助手: calculate_prepayment\n工具结果: 全部结清可节省利息12.5万。\n最近对话：用户: 那帮我约一下网点办理。",
        "用户画像：餐厅老板，经营贷50万5年，已还3年，受疫情影响收入下降。\n对话摘要：用户想申请展期。\n最近对话：用户: 我最近生意不好，月供压力大，能帮我看看展期吗？\n助手: 我先检查您的展期资格。",
        "用户画像：护士，月入1.1万，消费贷8万2年，已还10个月。\n对话摘要：用户想变更还款方式。\n工具操作：助手: calculate_repayment_method_switch\n工具结果: 变更后月供变化不大，建议维持现状。\n最近对话：用户: 那算了，还是继续等额本息吧。",
        "用户画像：公务员，公积金贷款90万30年，已还12年。\n对话摘要：用户想查询剩余还款计划。\n最近对话：用户: 帮我看一下还有多少期，每期还多少。\n助手: 我帮您生成还款计划表。",
        "用户画像：企业高管，月薪5万，房贷300万25年，已还2年。\n对话摘要：用户想一次性还清。\n最近对话：用户: 我有一笔投资回报到账了，想把房贷一次性还了。\n助手: 我先帮您算一下应还总额。",
        "用户画像：外卖骑手，车贷6万2年，逾期1次已还。\n对话摘要：用户忘了还款担心上征信。\n最近对话：用户: 我上个月忘了还车贷，晚了5天才还的，会上征信吗？\n助手: 一般30天以上才上报，但可能会有罚息，我帮您算一下。",
        "用户画像：律师，经营贷200万10年，已还3年。\n对话摘要：用户考虑展期。\n工具操作：助手: check_extension_eligibility\n工具结果: 资格通过。\n最近对话：用户: 展期后月供能少多少？帮我算算。",
        "用户画像：建筑工人，房贷40万20年，已还8年。\n对话摘要：用户想提前还部分本金。\n最近对话：用户: 我攒了10万，想提前还掉，是缩短年限还是减少月供？\n助手: 我帮您对比两种方案。",
        "用户画像：海员，房贷70万20年，已还3年，有6个月补还记录。\n对话摘要：用户担心之前逾期影响后续贷款。\n最近对话：用户: 我之前因为出海错过了还款，后来都补上了，这算严重逾期吗？\n助手: 补还后不算当前逾期，但历史记录会保留。",
    ]
}

# 扩展助手回复池
ASSISTANT_RESPONSES = {
    "ask_for_params": [
        "请问您的剩余本金是多少？",
        "请问贷款年利率是多少？",
        "请问已还了多少期？贷款总期数是多少？",
        "请问您想提前还多少金额？还是全部结清？",
        "请问您有无逾期记录？最近一次逾期是什么时候？",
        "请问您当前的还款方式是等额本息还是等额本金？",
        "请问您想改成哪种还款方式？",
        "请提供贷款合同编号或贷款账号。",
        "请问您的贷款类型是住房贷款、消费贷款还是经营贷款？",
        "请问您想延长多少个月？",
        "请问逾期本金是多少？逾期了多少天？",
        "请提供您的姓名和贷款编号，我帮您生成结清证明。",
        "请问您当前的月供是多少？",
        "请问您的贷款放款日期是哪天？",
        # 新增
        "请问您贷款的经办支行是哪家？",
        "请问您的提前还款金额来源是什么？（自有资金/其他贷款等）",
        "请确认您是否已预约提前还款？",
        "请问您配偶是否同意变更还款方式/展期？",
        "请提供您的收入证明或近6个月银行流水。",
    ],
    "provide_info": [
        "根据当前LPR 4.2%估算，提前还款可节省利息约15.3万。",
        "展期后月供将从5500元降至4200元，但总利息会增加。",
        "逾期罚息按合同利率的1.5倍计算，您需支付罚息85.6元。",
        "变更还款方式需支付200元手续费，但可节省利息约3.2万。",
        "还款计划表已生成，前3期月供中利息占比更高。",
        "结清证明需加盖银行公章，并携带至不动产登记中心办理解押。",
        "解押流程一般需要3-5个工作日，具体视当地登记中心而定。",
        "提前还款满一年免收违约金，不满一年按剩余本金的1%收取。",
        "展期后利率将适度上浮，一般为10个基点左右。",
        "逾期记录自还清之日起保留5年，建议保持良好的还款习惯。",
        "还款日每年可申请调整一次，需提前预约。",
        "续贷需要重新审批，银行会综合评估您的征信和收入。",
        # 新增
        "提前还款预约一般需要提前1个月向经办行提交书面申请。",
        "公积金冲还贷签约后，系统每月自动从公积金账户扣款，余额不足时再从银行卡补扣。",
        "抵押物价值下降可能触发银行要求补充抵押物或提前部分还款。",
        "贷款期间出售房产需先还清贷款或办理转按揭手续。",
        "贷款期间如遇利率调整，浮动利率贷款的重定价日一般为每年1月1日。",
    ],
    "tool_result": [
         "月供5300元，提前还20万后，缩短期限可节省利息18.2万。",
        "您当前符合展期条件，但展期后利率将上浮10个基点。",
        "逾期罚息共计235.8元，请在3个工作日内还清以免影响征信。",
        "从等额本息改为等额本金，首月月供将增加至6500元，总利息减少4.8万。",
        "已生成120期还款计划表，第1期月供6125元，最后一期月供4180元。",
        "结清证明模板已生成，请核对姓名和贷款编号后打印盖章。",
        "缩短期限方案：净节省12.8万；减少月供方案：净节省8.5万。",
        "展期36个月后月供降低1800元，但总利息增加3.2万。",
        "您的逾期罚息已计算，请尽快还款以免产生更多罚息并影响征信。",
        "变更方案已出：从等额本息改为等额本金，总利息减少4.8万，手续费200元。",
        # 新增
        "全部结清方案：应还总额65.8万，无违约金，可节省利息22.3万。",
        "展期申请因逾期记录被拒，建议先结清逾期款项，6个月后可重新申请。",
        "还款计划显示您已还了总利息的65%，剩余期限利息占比逐渐降低。",
        "结清证明已生成，请携带至不动产登记中心办理，记得先预约。",
        "提前还款对比：缩短期限比减少月供多节省5.6万利息，推荐前者。",
    ],
}

# 更多最近对话起始模板（可动态生成对话）
RECENT_CONV_STARTERS = [
    "用户: {user_input}\n助手: {assistant_response}",
    "用户: {user_input}\n助手: {assistant_response}\n用户: {follow_up}",
    "用户: {user_input}\n助手: {assistant_response}\n用户: {follow_up}\n助手: {follow_up_response}",
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
        "DIRECT_REPLY": 250,  # 原0.54 F1，急需增强
        "extension_management_skill": 200,  # 原0.63，需增加
        "check_extension_eligibility": 200,  # 原0.75，需增加
        "general_search_knowledge": 200,  # 原0.77，需增加
        "CLARIFY": 180,  # 0.80，适度增加
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
