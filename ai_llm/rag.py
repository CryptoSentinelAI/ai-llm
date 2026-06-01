"""
RAG 检索增强生成模块 — 文档切片、向量化、相似度检索。

轻量级实现，不依赖外部向量数据库。使用 numpy 做向量运算。

Examples:
    rag = RAG()
    rag.add_document("docs/readme.md")
    rag.add_text("Transformer 是一种基于自注意力机制的神经网络架构...")
    results = rag.query("什么是 Transformer？", top_k=3)
"""

import re
import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DocumentChunk:
    """文档切片"""
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({preview}...)"


class SimpleEmbedder:
    """简易向量化器

    使用字符级 n-gram 构建稀疏向量，零外部依赖。
    适合轻量场景和快速原型——生产环境请替换为 OpenAI Embeddings API。
    """

    def __init__(self, n: int = 3, dim: int = 256):
        self.n = n
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """将文本转为向量"""
        vec = np.zeros(self.dim, dtype=np.float32)

        # 字符级 n-gram 哈希
        text = text.lower()
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            idx = hash(ngram) % self.dim
            vec[idx] += 1.0

        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


class VectorStore:
    """内存向量存储"""

    def __init__(self):
        self._chunks: list[DocumentChunk] = []

    def add(self, chunk: DocumentChunk) -> None:
        self._chunks.append(chunk)

    def search(self, query_embedding: np.ndarray, top_k: int = 5
               ) -> list[tuple[DocumentChunk, float]]:
        """余弦相似度检索"""
        results = []
        for chunk in self._chunks:
            if chunk.embedding is not None:
                score = SimpleEmbedder.cosine_similarity(query_embedding, chunk.embedding)
                results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


class RAG:
    """RAG 检索增强生成

    支持从文件、文本、目录批量导入文档。
    自动分片 → 向量化 → 存储 → 检索。

    Args:
        chunk_size: 每片最大字符数
        chunk_overlap: 相邻切片重叠字符数
        embedder: 向量化器 (默认 SimpleEmbedder)

    Example:
        rag = RAG(chunk_size=512)
        rag.add_document("papers/attention.pdf")
        chunks = rag.query("自注意力机制的工作原理", top_k=3)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedder: Optional[SimpleEmbedder] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder or SimpleEmbedder()
        self.vector_store = VectorStore()

    def add_text(self, text: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        """添加文本并自动分片

        Args:
            text: 原始文本
            metadata: 附加元数据 (来源、页码等)

        Returns:
            创建的 DocumentChunk 列表
        """
        chunks = self._split_text(text, metadata or {})
        for chunk in chunks:
            chunk.embedding = self.embedder.embed(chunk.text)
            self.vector_store.add(chunk)
        return chunks

    def add_document(self, filepath: str) -> list[DocumentChunk]:
        """从文件导入文档

        支持 .txt, .md, .py, .json 等文本格式。

        Args:
            filepath: 文件路径

        Returns:
            创建的 DocumentChunk 列表
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        metadata = {"source": filepath, "filename": os.path.basename(filepath)}
        return self.add_text(text, metadata)

    def add_directory(self, dirpath: str, glob_pattern: str = "*.md"
                      ) -> list[DocumentChunk]:
        """批量导入目录下的文档

        Args:
            dirpath: 目录路径
            glob_pattern: 文件匹配模式

        Returns:
            所有创建的 DocumentChunk 列表
        """
        import glob
        all_chunks = []
        pattern = os.path.join(dirpath, glob_pattern)
        for filepath in glob.glob(pattern):
            try:
                chunks = self.add_document(filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"警告: 跳过 {filepath} — {e}")

        return all_chunks

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """检索最相关的文档片段

        Args:
            query: 查询文本
            top_k: 返回结果数
            min_score: 最低相似度阈值

        Returns:
            [(DocumentChunk, score), ...]
        """
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, top_k * 2)
        return [(c, s) for c, s in results if s >= min_score][:top_k]

    def query_texts(self, query: str, top_k: int = 5) -> list[str]:
        """检索结果仅返回文本列表"""
        return [chunk.text for chunk, _ in self.query(query, top_k)]

    def _split_text(self, text: str, metadata: dict) -> list[DocumentChunk]:
        """智能分片：优先按段落，长段落按句子，句子过长按字符"""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)

        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) <= self.chunk_size:
                current += ("\n\n" + para) if current else para
            else:
                if current:
                    chunks.append(DocumentChunk(text=current, metadata=dict(metadata)))
                current = para

                # 如果单个段落超过 chunk_size，按句子切分
                while len(current) > self.chunk_size:
                    split_point = self._find_split(current, self.chunk_size)
                    chunk_text = current[:split_point].strip()
                    if chunk_text:
                        chunks.append(DocumentChunk(text=chunk_text, metadata=dict(metadata)))
                    current = current[max(0, split_point - self.chunk_overlap):].strip()

        if current.strip():
            chunks.append(DocumentChunk(text=current.strip(), metadata=dict(metadata)))

        return chunks

    def _find_split(self, text: str, max_len: int) -> int:
        """找到最佳切分点：优先在句号、换行处切"""
        for pattern in [r'[。！？\n]', r'[，；：、]', r'\s']:
            matches = list(re.finditer(pattern, text[:max_len]))
            if matches:
                return matches[-1].end()
        return max_len

    def __len__(self) -> int:
        return len(self.vector_store)
