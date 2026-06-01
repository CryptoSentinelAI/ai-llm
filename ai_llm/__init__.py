"""
ai-llm: 轻量级 AI/LLM 工具库

多模型统一调用 · Prompt 模板引擎 · Token 管理 · RAG 检索增强
专为中文开发者优化。

Usage:
    from ai_llm import LLM, Prompt, RAG, TokenManager
"""

from ai_llm.llm import LLM, AsyncLLM
from ai_llm.prompt import Prompt, PromptTemplate, Conversation
from ai_llm.tokenizer import TokenManager, count_tokens
from ai_llm.rag import RAG, DocumentChunk, VectorStore

__version__ = "0.3.1"
__all__ = [
    "LLM", "AsyncLLM",
    "Prompt", "PromptTemplate", "Conversation",
    "TokenManager", "count_tokens",
    "RAG", "DocumentChunk", "VectorStore",
]
