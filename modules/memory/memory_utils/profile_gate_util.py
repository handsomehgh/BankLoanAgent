# author hgh
# version 1.0

import re
import logging
from config.models.memory_config import MemoryGateRules

logger = logging.getLogger(__name__)


class ProfileGate:
    """记忆门控：判断对话消息是否包含需要提取的用户画像信息"""

    def __init__(self, rules: MemoryGateRules):
        self.rules = rules
        # 预编译强信号正则
        self.strong_pattern = re.compile(
            '|'.join(f'(?:{p})' for p in rules.strong_patterns),
            re.IGNORECASE
        ) if rules.strong_patterns else None

        # 预编译显式触发器正则
        self.explicit_pattern = re.compile(
            '|'.join(f'(?:{t})' for t in rules.explicit_triggers),
            re.IGNORECASE
        ) if rules.explicit_triggers else None

        # 构建弱信号字典（小写单词 -> 分数）
        self.weak_signals: dict[str, int] = {}
        for item in rules.weak_signals:
            score = item.score
            for word in item.words:
                self.weak_signals[word.lower()] = score

        self.threshold = rules.match_threshold
        logger.info("ProfileGate 初始化完成，规则版本：%s", rules.version)

    def should_extract(self, messages) -> bool:
        """
        判断是否有足够信号需要提取画像。
        :param messages: 用户消息列表（字符串或带有 content 属性的消息对象）
        :return: True 表示应该执行提取
        """
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if not content:
                continue

            # 1. 显式更新指令（最高优先级）
            if self.explicit_pattern and self.explicit_pattern.search(content):
                return True

            # 2. 强信号匹配
            if self.strong_pattern and self.strong_pattern.search(content):
                return True

            # 3. 弱信号累积
            score = 0
            for word, point in self.weak_signals.items():
                if re.search(r'\b' + re.escape(word) + r'\b', content, re.IGNORECASE):
                    score += point
                    if score >= self.threshold:
                        return True

        return False
