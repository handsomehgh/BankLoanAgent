# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseResponseHandler(ABC):
    """响应处理器基类，所有工具专属处理器继承此类"""

    def __init__(self, result: Dict[str, Any]):
        self.result = result

    @abstractmethod
    def generate(self) -> str:
        """根据工具返回结果生成自然语言回复"""
        pass

    @staticmethod
    def _extract_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """安全提取数据，避免 KeyError"""
        return data if isinstance(data, dict) else {}