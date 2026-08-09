# 提示词库自检 + 版本字段校验：
# 防止"字面大括号未转义/占位符拼写错误"这类只在运行时暴露、且被 try/except 静默吞掉的模板缺陷。
# 校验对象：三个提示词库 yaml（config/rules/prompts_*.yaml），全部提示词的唯一存放地。
import string
from pathlib import Path

import pytest
import yaml
from langchain_core.prompts import ChatPromptTemplate

from config.models.prompt_library import PromptLibrary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_FILES = ("prompts_retrieval.yaml", "prompts_memory.yaml", "prompts_agent.yaml")


def _load_libraries():
    for name in LIBRARY_FILES:
        path = PROJECT_ROOT / "config" / "rules" / name
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # 顺带完成 pydantic 结构校验（text 与 system/human 二选一等）
        yield name, PromptLibrary(**data)


def _assert_valid_field(full_name: str, field_name: str):
    """占位符必须是合法标识符，否则说明示例中的字面大括号没有用 {{}} 转义"""
    root = field_name.split("[")[0].split(".")[0]
    assert root.isidentifier(), (
        f"提示词 {full_name} 含非法占位符 {field_name!r}，"
        f"字面大括号疑似未转义为 {{{{}}}}"
    )


def test_libraries_load_with_version():
    """每个库和每条提示词都必须带版本号（版本归因的前提）"""
    total = 0
    for lib_name, library in _load_libraries():
        assert library.version, f"{lib_name} 缺少库版本号"
        assert library.prompts, f"{lib_name} 为空库"
        for key, entry in library.prompts.items():
            assert entry.version, f"{lib_name}:{key} 缺少版本号"
            total += 1
    assert total >= 30, f"提示词总数异常：{total}，疑似迁移遗漏"


def test_text_entries_are_renderable():
    """text 形态条目按 str.format 规则干跑渲染"""
    for lib_name, library in _load_libraries():
        for key, entry in library.prompts.items():
            if entry.text is None:
                continue
            full_name = f"{lib_name}:{key}"
            fields = []
            try:
                parsed = list(string.Formatter().parse(entry.text))
            except ValueError as e:
                pytest.fail(f"提示词 {full_name} 大括号语法错误（字面大括号需写成 {{{{}}}}）: {e}")
            for _, field_name, _, _ in parsed:
                if field_name is None:
                    continue
                assert field_name != "", f"提示词 {full_name} 含空占位符 {{}}，str.format 会要求位置参数"
                _assert_valid_field(full_name, field_name)
                fields.append(field_name)
            if fields:
                entry.text.format(**{f: "test" for f in fields})


def test_chat_entries_are_renderable():
    """system(+human) 形态条目按 langchain f-string 规则干跑渲染"""
    for lib_name, library in _load_libraries():
        for key, entry in library.prompts.items():
            if entry.text is not None:
                continue
            full_name = f"{lib_name}:{key}"
            messages = [("system", entry.system)]
            if entry.human:
                messages.append(("human", entry.human))
            template = ChatPromptTemplate.from_messages(messages)
            for var in template.input_variables:
                _assert_valid_field(full_name, var)
            # 用假变量干跑一次渲染，确保模板无残留语法错误
            template.invoke({var: "test" for var in template.input_variables})
