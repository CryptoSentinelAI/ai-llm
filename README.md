# ai-llm

<p align="center">
  <b>轻量级 AI/LLM 工具库</b><br>
  多模型调用 · Prompt 编排 · Token 管理 · RAG 检索增强<br>
  专为中文开发者优化
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.9-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.3.1-orange" alt="Version">
  <img src="https://github.com/CryptoSentinelAI/ai-llm/actions/workflows/test.yml/badge.svg" alt="CI">
</p>

---

## 简介

`ai-llm` 是一个面向中文开发者的轻量级 AI/LLM 工具库，封装了日常 AI 开发中最常用的四个模块：

| 模块 | 功能 |
|---|---|
| **LLM** | 多模型统一调用（OpenAI / 兼容 API），同步+异步，自动重试，流式输出 |
| **Prompt** | 模板引擎，变量注入，链式调用，多轮对话管理 |
| **Token** | 精确计数（tiktoken），成本估算，预算控制，长文本截断 |
| **RAG** | 文档切片，向量化，相似度检索，零外部依赖 |

## 快速开始

```bash
pip install ai-llm
```

```python
from ai_llm import LLM, PromptTemplate, TokenManager, RAG

# 多模型调用
llm = LLM(model="gpt-4o")
response = llm.chat("你好，世界！")

# 使用第三方兼容 API
llm = LLM(model="deepseek-chat", base_url="https://api.deepseek.com/v1")
response = llm.chat("解释量子计算")

# Prompt 模板
tpl = PromptTemplate("将以下{source}翻译为{target}：{text}",
                     defaults={"source": "中文"})
prompt = tpl.format(target="English", text="人工智能")

# Token 管理
tm = TokenManager("gpt-4o")
cost = tm.estimate_cost(input_tokens=1000, output_tokens=500)
print(f"成本: ${cost.total_cost:.4f}")

# RAG 检索
rag = RAG(chunk_size=256)
rag.add_text("Transformer 是一种基于自注意力机制的深度学习架构...")
results = rag.query("什么是自注意力机制？", top_k=3)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk.text[:60]}...")
```

## 模块详解

### LLM — 多模型统一调用

```python
from ai_llm import LLM, AsyncLLM

# 同步调用
with LLM(model="gpt-4o") as llm:
    resp = llm.chat("你好")
    print(resp.content)

# 流式输出
for chunk in llm.stream("讲一个笑话"):
    print(chunk, end="")

# 异步调用
async with AsyncLLM(model="gpt-4o") as llm:
    resp = await llm.chat("你好")
```

### Prompt — 模板引擎

```python
from ai_llm import PromptTemplate, Conversation, Prompt

# 变量注入
tpl = PromptTemplate("角色：{role}\n任务：{task}")
result = tpl.format(role="翻译官", task="翻译以下文本")

# 多轮对话
conv = Conversation(system="你是助手")
conv.add_user("问题1")
conv.add_assistant("回答1")
conv.add_user("追问")
messages = conv.to_messages()

# 链式调用
chain = Prompt.chain(
    PromptTemplate("分析代码：{code}"),
    PromptTemplate("给出修复建议：{prev}"),
)
```

### Token — 管理与成本

```python
from ai_llm import TokenManager, count_tokens

tm = TokenManager("gpt-4o", budget=0.05)
tokens = tm.count("你好世界 Hello World")

# 消息列表批量计数
messages = [{"role": "user", "content": "你好"}]
total = tm.count_messages(messages)

# 成本估算 + 预算控制
usage = tm.estimate_cost(input_tokens=1000, output_tokens=500)
if tm.check_budget(usage.total_cost):
    print("在预算内")

# 长文本截断
truncated = tm.truncate("一段很长的文本..." * 100, max_tokens=500)
```

### RAG — 检索增强

```python
from ai_llm import RAG

rag = RAG(chunk_size=512, chunk_overlap=50)

# 从文件导入
rag.add_document("docs/knowledge.md")

# 批量导入目录
rag.add_directory("docs/", "*.md")

# 检索
results = rag.query("什么是 Transformer？", top_k=3)
texts = rag.query_texts("RAG 原理")  # 仅返回文本
```

## 支持的模型

| 模型 | Token 计数 | 定价 |
|---|---|---|
| gpt-4o | ✅ | ✅ |
| gpt-4o-mini | ✅ | ✅ |
| gpt-4-turbo | ✅ | ✅ |
| gpt-3.5-turbo | ✅ | ✅ |
| claude-3.5-sonnet | ✅ | ✅ |
| claude-3-haiku | ✅ | ✅ |
| deepseek-chat | ✅ | ✅ |

## 安装

```bash
# PyPI
pip install ai-llm

# 开发者安装
git clone https://github.com/CryptoSentinelAI/ai-llm.git
cd ai-llm
pip install -e ".[dev]"
```

## 运行测试

```bash
pytest tests/ -v
```

## 示例

```bash
python examples/basic_usage.py
```

## 适用场景

- AI 应用快速原型开发
- 企业内部 AI 工具链搭建
- 中文 NLP 任务处理
- 多模型对比测试与选型
- Token 成本敏感的 API 调用优化

## 贡献

欢迎提交 Issue 和 PR！详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 许可证

Apache License 2.0 © CryptoSentinelAI
