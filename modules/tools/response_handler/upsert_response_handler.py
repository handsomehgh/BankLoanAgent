# author hgh
# version 1.0
from typing import Dict, Any

from modules.tools.response_handler.base_response_handler import BaseResponseHandler


class UpsertLoanInterestHandler(BaseResponseHandler):
    def __init__(self, result: Dict[str, Any], config: Dict[str, str]):
        super().__init__(result)
        self.config = config

    def generate(self) -> str:
        signal = self.result.get('signal', '')
        data = self._extract_data(self.result.get('data', {}))

        template = self.config.get(signal, '')
        if not template:
            return ""

        # 准备模板变量
        loan_type = data.get('loan_type', '贷款')
        method = data.get('repayment_method', '')
        status = data.get('status', '')
        current_status = data.get('current_status', '处理中')

        # 构建模板变量映射
        variables = {
            'loan_type': loan_type,
            'desired_amount': data.get('desired_amount', 0),
            'term_years': data.get('term_years', 0),
            'application_no': data.get('application_no', ''),
            'repayment_method': f"，{method}" if method else "",
            'status_hint': self._get_status_hint(signal, status, current_status),
        }

        return template.format(**variables)

    def _get_status_hint(self, signal: str, status: str, current_status: str) -> str:
        """根据信号和状态生成状态提示文本"""
        if signal == 'updated':
            if status == '待处理':
                return "将重新进入处理流程，客户经理会按最新需求与您联系。"
            return ""
        if signal == 'need_confirm':
            if current_status == '处理中':
                return "已有工作人员在处理中"
            return "之前的意向已处理完毕"
        return ""
