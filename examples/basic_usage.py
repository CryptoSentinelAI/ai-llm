"""
ai-llm 使用示例

运行: python examples/basic_usage.py
"""

from ai_llm import PromptTemplate, Conversation, TokenManager, RAG


def demo_prompt():
    print("=" * 50)
    print("1. Prompt 模板引擎")
    print("=" * 50)

    tpl = PromptTemplate(
        "将以下{source_lang}文本翻译为{target_lang}：\n{text}",
        defaults={"source_lang": "中文"}
    )
    result = tpl.format(target_lang="English", text="人工智能正在改变世界")
    print(f"  模板: {tpl}")
    print(f"  变量: {tpl.fields}")
    print(f"  结果: {result}")
    print()


def demo_conversation():
    print("=" * 50)
    print("2. 多轮对话管理")
    print("=" * 50)

    conv = Conversation(system="你是一个专业的翻译助手，只输出翻译结果。")
    conv.add_user("翻译以下内容为英文：今天天气真好")
    conv.add_assistant("The weather is really nice today.")
    conv.add_user("再把这句话翻译为日文")

    msgs = conv.to_messages()
    print(f"  对话轮数: {len(msgs)}")
    for msg in msgs:
        role = msg["role"]
        content = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
        print(f"  [{role}] {content}")
    print()


def demo_token():
    print("=" * 50)
    print("3. Token 管理与成本估算")
    print("=" * 50)

    tm = TokenManager("gpt-4o")
    text = "你好世界！Hello World！AI 正在改变我们的生活方式。"
    tokens = tm.count(text)
    cost = tm.estimate_cost(input_tokens=tokens, output_tokens=200)

    print(f"  模型: {tm.model}")
    print(f"  文本: {text}")
    print(f"  Token 数: {tokens}")
    print(f"  成本估算: ${cost.total_cost:.6f} (输入: ${cost.input_cost:.6f}, 输出: ${cost.output_cost:.6f})")
    print()

    # 预算控制
    budget_tm = TokenManager("gpt-4o", budget=0.05)
    large_usage = budget_tm.estimate_cost(input_tokens=50000, output_tokens=10000)
    within_budget = budget_tm.check_budget(large_usage.total_cost)
    print(f"  预算: $0.05, 使用: ${large_usage.total_cost:.4f}, 通过: {within_budget}")


def demo_rag():
    print("\n" + "=" * 50)
    print("4. RAG 检索增强生成")
    print("=" * 50)

    rag = RAG(chunk_size=256)

    # 添加知识库
    documents = [
        "Transformer 架构于 2017 年在论文 'Attention Is All You Need' 中首次提出。"
        "它完全基于自注意力（Self-Attention）机制，摒弃了传统的循环神经网络结构。"
        "Transformer 已成为现代大语言模型的基础架构，GPT、BERT 等模型均基于此。",

        "GPT（Generative Pre-trained Transformer）是由 OpenAI 开发的系列大语言模型。"
        "从 GPT-1 到 GPT-4，模型规模和能力不断提升。GPT-4 支持多模态输入，"
        "能够理解和生成自然语言文本，在多种基准测试中表现优异。",

        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的 AI 技术。"
        "它先从外部知识库检索相关文档，再基于检索结果生成回答。"
        "这种方法有效减少了幻觉（Hallucination），提高了回答的准确性和可靠性。",
    ]

    for i, doc in enumerate(documents):
        rag.add_text(doc, {"id": i + 1})

    print(f"  文档切片数: {len(rag)}")
    print()

    # 检索
    queries = [
        "Transformer 是什么时候提出的？",
        "RAG 如何减少幻觉？",
    ]

    for query in queries:
        print(f"  查询: {query}")
        results = rag.query(query, top_k=2)
        for i, (chunk, score) in enumerate(results):
            preview = chunk.text[:80].replace("\n", " ")
            print(f"    [{i+1}] score={score:.3f} | {preview}...")
        print()


if __name__ == "__main__":
    demo_prompt()
    demo_conversation()
    demo_token()
    demo_rag()
    print("=" * 50)
    print("所有示例运行完成！")
