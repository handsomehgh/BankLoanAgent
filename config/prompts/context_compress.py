from langchain_core.prompts import ChatPromptTemplate

CONTEXT_COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息检索专家。请根据用户问题，对候选文档进行相关性排序。
要求：
- 只返回排序后的文档ID列表，格式为：[id1, id2, id3, ...]
- 文档越相关，排在越前面
- 不要输出任何解释、说明或额外文字

示例：
用户问题：房贷利率一般是多少？
候选文档：
[id:0] 当前五年期以上LPR为4.2%，首套房利率不低于LPR+60BP。
[id:1] 申请房贷需要提供身份证、收入证明、银行流水等材料。
[id:2] 提前还款可以在还款满一年后申请，违约金为剩余本金的1%。
[id:3] 根据最新政策，首套房贷款利率最低可至4.00%。
输出：[3, 0, 1, 2]"""),
    ("human", """用户问题：{query}
候选文档：
{docs}

输出：""")
])