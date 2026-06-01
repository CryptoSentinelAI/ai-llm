"""
多模型统一调用模块。

支持 OpenAI、兼容 API 的第三方模型、以及本地模型端点。
统一接口设计，一行代码切换模型。

Examples:
    llm = LLM(model="gpt-4o")
    response = llm.chat("你好")

    llm = LLM(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
    response = llm.chat("解释量子计算", temperature=0.3)
"""

import os
import json
import time
from typing import Optional, Any, Literal
from dataclasses import dataclass, field

import httpx


@dataclass
class LLMResponse:
    """LLM 统一响应格式"""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    def __str__(self) -> str:
        return self.content


class LLM:
    """多模型统一调用客户端

    支持通过 base_url 对接任何 OpenAI-compatible API。
    自动处理重试、超时、流式输出。

    Args:
        model: 模型名称 (默认 gpt-4o)
        api_key: API 密钥 (默认读环境变量 OPENAI_API_KEY)
        base_url: API 基础地址 (可选，用于第三方兼容接口)
        max_retries: 最大重试次数
        timeout: 请求超时秒数
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.max_retries = max_retries
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "请设置 OPENAI_API_KEY 环境变量，或传入 api_key 参数"
            )

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """发送单轮对话

        Args:
            prompt: 用户消息
            system: 系统提示词
            temperature: 温度参数 (0-2)
            max_tokens: 最大输出 token 数
            top_p: nucleus sampling
            stop: 停止词列表

        Returns:
            LLMResponse 对象
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._request(messages, temperature, max_tokens, top_p, stop)

    def chat_with_history(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """多轮对话

        Args:
            messages: 消息列表 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            top_p: nucleus sampling

        Returns:
            LLMResponse 对象
        """
        return self._request(messages, temperature, max_tokens, top_p)

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """流式输出（生成器）

        Yields:
            每次 yield 一段文本
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        for attempt in range(self.max_retries):
            try:
                with self._client.stream("POST", "/chat/completions", json=body) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                return
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                return
            except httpx.HTTPError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _request(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop:
            body["stop"] = stop

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post("/chat/completions", json=body)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason", ""),
                    raw=data,
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except httpx.HTTPError as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise last_error  # type: ignore

    def close(self):
        """关闭 HTTP 客户端"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncLLM(LLM):
    """异步版 LLM 客户端，接口与 LLM 一致"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
        )

    async def chat(self, prompt: str, system: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self._async_request(messages, kwargs.get("temperature", 0.7),
                                          kwargs.get("max_tokens", 4096))

    async def _async_request(self, messages, temperature, max_tokens) -> LLMResponse:
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        for attempt in range(self.max_retries):
            try:
                resp = await self._async_client.post("/chat/completions", json=body)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason", ""),
                    raw=data,
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    await __import__("asyncio").sleep(2 ** attempt)
                    continue
                raise
            except httpx.HTTPError:
                if attempt == self.max_retries - 1:
                    raise
                await __import__("asyncio").sleep(2 ** attempt)

    async def close(self):
        await self._async_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
