# author hgh
# version 1.0
from langchain_core.prompts import ChatPromptTemplate

EVIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """根据以下最近几轮用户发言，判断当前提取的用户画像内容属于哪种证据类型：
可选类型：{valid_types}

最近对话：
{conversation}

请仅输出证据类型（如 explicit_statement），不要解释。""")
])
