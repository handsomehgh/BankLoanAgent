# author hgh
# version 1.0
DETECT_SENTIMENT_PROMPT = """分析以下对话中客户的情绪倾向，仅输出一个代码。

情绪选项：
- positive：积极、满意、感谢
- neutral：中性、普通咨询
- anxious：焦虑、着急、紧张
- frustrated：沮丧、失望、愤怒

对话内容：
{text}

情绪代码："""