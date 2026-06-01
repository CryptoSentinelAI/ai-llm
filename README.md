# ai-llm

<p align="center">
  <b>轻量级 AI/LLM 工具库</b><br>
  多模型调用 · Prompt 编排 · Token 管理 · RAG 检索增强<br>
  专为中文开发者优化
</p>

---

## 简介

`ai-llm` 是一个面向中文开发者的轻量级 AI/LLM 工具库，封装了多模型调用、Prompt 编排、Token 管理、RAG 检索增强等常用功能，让你用最少的代码完成最常用的 AI 任务。

## 特性

- **多模型统一调用** — 支持 OpenAI、Claude、本地模型，统一 API 切换
- **Prompt 编排引擎** — 模板化 Prompt 管理，变量注入，链式调用
- **Token 管理** — 自动计数、截断、成本估算，控制预算
- **RAG 检索增强** — 文档切片、向量化、相似度检索，开箱即用
- **中文优化** — 中文分词、提示词模板全部针对中文场景调优

## 快速开始

```python
from ai_llm import LLM, Prompt, RAG

# 多模型调用
llm = LLM(model="gpt-4o")
response = llm.chat("你好，世界！")

# Prompt 编排
prompt = Prompt.from_template("将以下文本翻译为英文：{text}")
result = prompt.run(text="人工智能正在改变世界")

# RAG 检索
rag = RAG.from_documents("docs/")
answer = rag.query("什么是Transformer架构？")
```

## 安装

```bash
pip install ai-llm
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
