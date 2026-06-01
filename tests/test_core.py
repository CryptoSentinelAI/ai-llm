"""ai-llm 测试套件"""
import os
import tempfile
import pytest
from ai_llm.prompt import PromptTemplate, Conversation, Prompt, ChainPipeline
from ai_llm.tokenizer import TokenManager, count_tokens, TokenUsage
from ai_llm.rag import RAG, DocumentChunk, SimpleEmbedder, VectorStore


class TestPromptTemplate:
    def test_basic_format(self):
        tpl = PromptTemplate("你好，{name}！")
        assert tpl.format(name="世界") == "你好，世界！"

    def test_missing_field_raises(self):
        tpl = PromptTemplate("你好，{name}！")
        with pytest.raises(KeyError, match="缺少模板变量"):
            tpl.format()

    def test_defaults(self):
        tpl = PromptTemplate("将以下内容翻译为{lang}：{text}",
                            defaults={"lang": "英文"})
        result = tpl.format(text="你好")
        assert "英文" in result
        assert "你好" in result

    def test_extract_fields(self):
        tpl = PromptTemplate("{a} + {b} = {c}")
        assert tpl.fields == {"a", "b", "c"}

    def test_extra_kwargs_ignored(self):
        tpl = PromptTemplate("{name}")
        result = tpl.format(name="AI", extra="ignored")
        assert result == "AI"


class TestConversation:
    def test_basic(self):
        conv = Conversation(system="你是一个助手")
        conv.add_user("你好")
        conv.add_assistant("你好！")
        msgs = conv.to_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_clear_keep_system(self):
        conv = Conversation(system="系统")
        conv.add_user("问题1")
        conv.clear(keep_system=True)
        msgs = conv.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_clear_discard_all(self):
        conv = Conversation(system="系统")
        conv.add_user("问题1")
        conv.clear(keep_system=False)
        assert len(conv.to_messages()) == 0


class TestChainPipeline:
    def test_build(self):
        t1 = PromptTemplate("分析：{text}")
        t2 = PromptTemplate("修复：{prev}")
        pipeline = ChainPipeline((t1, t2))
        prompts = pipeline.build(text="def foo() return 1")
        assert len(prompts) == 2
        assert "分析" in prompts[0]
        assert "修复" in prompts[1]


class TestTokenManager:
    def test_count_basic(self):
        tm = TokenManager("gpt-4o")
        tokens = tm.count("Hello World")
        assert tokens > 0

    def test_chinese_text(self):
        tm = TokenManager("gpt-4o")
        tokens = tm.count("你好世界")
        assert tokens > 0

    def test_count_messages(self):
        tm = TokenManager("gpt-4o")
        messages = [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
        ]
        total = tm.count_messages(messages)
        assert total > 0

    def test_estimate_cost(self):
        tm = TokenManager("gpt-4o")
        usage = tm.estimate_cost(input_tokens=1000, output_tokens=500)
        assert usage.total_tokens == 1500
        assert usage.total_cost > 0

    def test_budget_check(self):
        tm = TokenManager("gpt-4o", budget=0.01)
        assert tm.check_budget(0.005) is True
        assert tm.check_budget(0.02) is False

    def test_list_models(self):
        models = TokenManager.list_models()
        assert "gpt-4o" in models
        assert len(models) > 3

    def test_count_tokens_convenience(self):
        tokens = count_tokens("Hello", "gpt-4o")
        assert tokens > 0


class TestRAG:
    def test_add_and_query(self):
        rag = RAG(chunk_size=256)
        rag.add_text("Transformer 是一种基于自注意力机制的深度学习架构。"
                     "它广泛应用于自然语言处理任务。")
        results = rag.query("自注意力机制", top_k=2)
        assert len(results) > 0

    def test_chinese_split(self):
        rag = RAG(chunk_size=100)
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = rag.add_text(text)
        assert len(chunks) >= 1

    def test_min_score_filter(self):
        rag = RAG(chunk_size=256)
        rag.add_text("Python 是一门编程语言。")
        results = rag.query("完全不相关的内容", top_k=5, min_score=0.8)
        assert len(results) == 0

    def test_query_texts(self):
        rag = RAG(chunk_size=256)
        rag.add_text("机器学习是人工智能的一个分支。")
        texts = rag.query_texts("机器学习", top_k=1)
        assert len(texts) == 1
        assert "机器学习" in texts[0]

    def test_empty_rag(self):
        rag = RAG()
        assert len(rag) == 0
        results = rag.query("任意查询")
        assert results == []

    def test_vector_store(self):
        store = VectorStore()
        embedder = SimpleEmbedder()
        c1 = DocumentChunk(text="A", embedding=embedder.embed("A"))
        c2 = DocumentChunk(text="B", embedding=embedder.embed("B"))
        store.add(c1)
        store.add(c2)
        assert len(store) == 2


class TestSimpleEmbedder:
    def test_embed_output_shape(self):
        embedder = SimpleEmbedder(dim=128)
        vec = embedder.embed("测试文本")
        assert vec.shape == (128,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 0.001

    def test_cosine_similarity(self):
        import numpy as np
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        assert SimpleEmbedder.cosine_similarity(a, b) == 0.0
        assert abs(SimpleEmbedder.cosine_similarity(a, c) - 1.0) < 0.001
