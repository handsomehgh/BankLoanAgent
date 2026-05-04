# author hgh
# version 1.0
DETECT_EVIDENCE_PROMPT = """根据以下对话，判断客户提供的收入或资产证据类型。

证据类型选项（仅输出代码）：
- bank_statement：提及银行流水、工资单、代发记录
- credit_report：提及征信报告、信用评分
- tax_document：提及税单、个税完税证明
- explicit_statement：仅口头陈述，无文档佐证

对话内容：
{conversation}

证据类型代码："""