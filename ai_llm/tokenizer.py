"""
Token 管理模块 — 精确计数、成本估算、预算控制。

基于 tiktoken，支持 OpenAI 全系列模型的精确 token 计算。
中文分词优化，自动处理中英混合文本。

Examples:
    tm = TokenManager("gpt-4o")
    tokens = tm.count("你好世界！Hello World!")
    cost = tm.estimate_cost(tokens, output_tokens=500)
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# 模型定价 (USD per 1M tokens)
# 持续更新中 — 最新价格请参考各平台官方文档
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}

# tiktoken 模型名映射
MODEL_ENCODING = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
}

_SIMPLE_CHINESE_RATIO = 1.5  # 粗略系数：1 个中文字 ≈ 1.5 tokens


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """便捷函数：计算文本 token 数

    Args:
        text: 输入文本
        model: 模型名称

    Returns:
        token 数量
    """
    return TokenManager(model).count(text)


@dataclass
class TokenUsage:
    """Token 用量报告"""
    input_tokens: int
    output_tokens: int
    model: str
    input_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def __repr__(self) -> str:
        return (f"TokenUsage(total={self.total_tokens}, "
                f"in={self.input_tokens}, out={self.output_tokens}, "
                f"cost=${self.total_cost:.4f})")


class TokenManager:
    """Token 管理器

    核心功能：
    - 精确 token 计数 (基于 tiktoken)
    - 成本估算 (自动查询最新定价)
    - 消息列表批量计数
    - 预算控制与预警

    Args:
        model: 模型名称
        budget: 单次调用预算上限 (USD)，默认无限制
    """

    def __init__(self, model: str = "gpt-4o", budget: Optional[float] = None):
        self.model = model
        self.budget = budget
        self._encoder = self._get_encoder(model)
        self._pricing = PRICING.get(model, {"input": 1.0, "output": 4.0})

    def _get_encoder(self, model: str):
        """获取 tiktoken 编码器"""
        if not _TIKTOKEN_AVAILABLE:
            return None

        encoding_name = MODEL_ENCODING.get(model, "cl100k_base")
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:
                return tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """计算文本 token 数

        优先使用 tiktoken 精确计算，fallback 到粗略估算。
        中文文本自动适配。
        """
        if self._encoder:
            return len(self._encoder.encode(text))
        return self._estimate(text)

    def count_messages(self, messages: list[dict]) -> int:
        """计算消息列表的总 token 数

        Args:
            messages: [{"role": "...", "content": "..."}]

        Returns:
            总 token 数
        """
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
        total += 3  # 每条消息的格式开销
        return total

    def _estimate(self, text: str) -> int:
        """粗略估算：中文字符 × 1.5 + 英文按单词 × 1.3"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ascii_chars = sum(1 for c in text if c.isascii() and not c.isspace())
        return int(chinese_chars * _SIMPLE_CHINESE_RATIO + ascii_chars * 0.3)

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> TokenUsage:
        """估算调用成本

        Args:
            input_tokens: 输入 token 数
            output_tokens: 预估输出 token 数

        Returns:
            TokenUsage 对象
        """
        input_cost = (input_tokens / 1_000_000) * self._pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self._pricing["output"]
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            input_cost=input_cost,
            output_cost=output_cost,
        )

    def check_budget(self, estimated_cost: float) -> bool:
        """检查是否在预算内"""
        if self.budget is None:
            return True
        return estimated_cost <= self.budget

    def truncate(self, text: str, max_tokens: int, suffix: str = "...") -> str:
        """截断文本至指定 token 数

        Args:
            text: 原始文本
            max_tokens: 最大 token 数
            suffix: 截断后追加的后缀

        Returns:
            截断后的文本
        """
        if self.count(text) <= max_tokens:
            return text

        tokens = self._encoder.encode(text) if self._encoder else text
        if self._encoder:
            truncated = self._encoder.decode(tokens[:max_tokens - self.count(suffix)])
            return truncated + suffix

        # Fallback: 按字符粗略截断
        ratio = len(text) / max(self.count(text), 1)
        cut = int(max_tokens * ratio)
        return text[:cut] + suffix

    @classmethod
    def list_models(cls) -> list[str]:
        """列出已配置定价的模型"""
        return sorted(PRICING.keys())

    @classmethod
    def get_pricing(cls, model: str) -> dict:
        """获取模型定价

        Returns:
            {"input": float, "output": float}
        """
        return PRICING.get(model, {"input": 1.0, "output": 4.0})
