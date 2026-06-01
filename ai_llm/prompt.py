"""
Prompt 模板引擎 — 变量注入、链式调用、对话管理。

支持模板变量、条件渲染、多轮对话编排。

Examples:
    # 基础模板
    tpl = PromptTemplate("将以下文本翻译为{target_lang}：{text}")
    prompt = tpl.format(target_lang="英文", text="你好世界")

    # 链式调用
    pipeline = Prompt.chain(
        PromptTemplate("分析以下代码的问题：{code}"),
        PromptTemplate("基于分析结果，给出修复建议：{prev}"),
    )
    result = pipeline.run(code="def foo() return 1")
"""

import re
from typing import Any, Optional
from dataclasses import dataclass, field


class PromptTemplate:
    """Prompt 模板 — 支持变量占位符

    Args:
        template: 模板字符串，使用 {var_name} 占位
        defaults: 变量默认值

    Example:
        tpl = PromptTemplate("翻译：{text}", defaults={"target_lang": "英文"})
        result = tpl.format(text="你好")
    """

    def __init__(self, template: str, defaults: Optional[dict[str, Any]] = None):
        self.template = template
        self.defaults = defaults or {}
        self._fields = self._extract_fields(template)

    def _extract_fields(self, template: str) -> set[str]:
        return set(re.findall(r"\{(\w+)\}", template))

    def format(self, **kwargs) -> str:
        """填充模板变量

        Args:
            **kwargs: 变量键值对

        Returns:
            填充后的字符串

        Raises:
            KeyError: 缺少必需变量且无默认值
        """
        values = {**self.defaults, **kwargs}
        missing = self._fields - set(values.keys())
        if missing:
            raise KeyError(f"缺少模板变量: {missing}")
        return self.template.format(**{k: values[k] for k in self._fields})

    @property
    def fields(self) -> set[str]:
        """返回模板中的所有变量名"""
        return self._fields

    def __repr__(self) -> str:
        return f"PromptTemplate(fields={self._fields})"


class Conversation:
    """多轮对话管理器

    自动维护对话历史，追加系统提示词。

    Example:
        conv = Conversation(system="你是一个翻译助手")
        conv.add_user("你好")
        conv.add_assistant("你好！有什么可以帮你？")
        conv.add_user("翻译：hello")
        messages = conv.to_messages()
    """

    def __init__(self, system: Optional[str] = None):
        self._messages: list[dict[str, str]] = []
        if system:
            self._messages.append({"role": "system", "content": system})

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def to_messages(self) -> list[dict[str, str]]:
        """返回标准 messages 格式"""
        return list(self._messages)

    def clear(self, keep_system: bool = True) -> None:
        """清空对话历史"""
        if keep_system and self._messages and self._messages[0]["role"] == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"Conversation(turns={len(self._messages)})"


class Prompt:
    """高级 Prompt 管理器

    静态方法集合，用于链式调用、批量处理等高级场景。

    Example:
        # 链式 Prompt
        prompts = [
            PromptTemplate("总结以下内容：{text}"),
            PromptTemplate("将以下总结翻译为英文：{prev}"),
        ]
        result = Prompt.chain(*prompts).run(text="这是一段很长的中文文章...")
    """

    @staticmethod
    def chain(*templates: PromptTemplate) -> "ChainPipeline":
        """创建链式 Prompt 管道

        每个模板的输出作为下一个模板的 {prev} 变量。

        Args:
            *templates: 一系列 PromptTemplate 对象

        Returns:
            ChainPipeline 对象
        """
        return ChainPipeline(templates)

    @staticmethod
    def from_file(path: str) -> PromptTemplate:
        """从文件加载 Prompt 模板"""
        with open(path, "r", encoding="utf-8") as f:
            return PromptTemplate(f.read().strip())

    @staticmethod
    def batch_format(template: PromptTemplate, rows: list[dict]) -> list[str]:
        """批量格式化

        Args:
            template: 模板对象
            rows: 变量字典列表

        Returns:
            格式化后的字符串列表
        """
        return [template.format(**row) for row in rows]


class ChainPipeline:
    """链式 Prompt 管道

    依次执行模板，每个输出作为下一个模板的 {prev} 变量。
    需要配合 LLM 实现实际调用——本类仅做 Prompt 串联。
    """

    def __init__(self, templates: tuple[PromptTemplate, ...]):
        self.templates = templates

    def build(self, **inputs) -> list[str]:
        """构建所有步骤的 Prompt（不执行 LLM 调用）

        Returns:
            每步的 Prompt 字符串列表
        """
        prompts = []
        current = inputs
        prev_output = ""

        for tpl in self.templates:
            vars_with_prev = {**current, "prev": prev_output}
            prompt = tpl.format(**{k: v for k, v in vars_with_prev.items()
                                   if k in tpl.fields})
            prompts.append(prompt)
            prev_output = f"[上一步输出将在此插入]"

        return prompts

    def run_with(self, llm_callable, **inputs) -> list[str]:
        """执行链式调用

        Args:
            llm_callable: 可调用对象，接受 prompt 字符串返回响应字符串
            **inputs: 初始输入变量

        Returns:
            每步的 LLM 响应列表
        """
        results = []
        prev_output = ""

        for tpl in self.templates:
            vars_with_prev = {**inputs, "prev": prev_output}
            prompt = tpl.format(**{k: v for k, v in vars_with_prev.items()
                                   if k in tpl.fields})
            prev_output = llm_callable(prompt)
            results.append(prev_output)

        return results
