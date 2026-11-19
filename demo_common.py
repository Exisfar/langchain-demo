"""各阶段共用的配置、文本切分和索引读写。"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "storage"
INDEX_PATH = STORAGE / "index.json"
DATABASE_PATH = STORAGE / "conversations-python.sqlite"
EMBEDDING_MODEL = "text-embedding-3-small"

# 从项目目录加载，运行命令时不必手动激活环境或导出变量。
load_dotenv(ROOT / ".env")


def chat_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("请在 .env 中设置 OPENAI_MODEL")
    return ChatOpenAI(model=model, use_responses_api=True)


def embedding_model(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


def shop_documents() -> list[Document]:
    """读取并切分 Markdown，保留文件来源与片段编号。"""
    text = (ROOT / "knowledge/shop.md").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    documents = splitter.create_documents(
        [text], metadatas=[{"source": "knowledge/shop.md"}]
    )
    for index, document in enumerate(documents, start=1):
        document.metadata["chunk"] = index
    return documents


def format_context(documents: list[Document]) -> str:
    parts = []
    for document in documents:
        source = document.metadata.get("source", "未知来源")
        if "chunk" in document.metadata:
            source += f"#片段{document.metadata['chunk']}"
        parts.append(f"[{source}]\n{document.page_content}")
    return "\n\n".join(parts)


def save_index(
    documents: list[Document],
    embeddings: Embeddings,
    path: Path | None = None,
    model: str = EMBEDDING_MODEL,
) -> None:
    path = path if path is not None else INDEX_PATH
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    data = {
        "embedding_model": model,
        "documents": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ],
        "vectors": vectors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class CachedDocumentEmbeddings(Embeddings):
    """恢复索引时使用已保存的文档向量，问题仍由真实 Embedding 模型处理。"""

    def __init__(self, documents, vectors, query_embeddings: Embeddings):
        self._vectors = {
            doc.page_content: vector
            for doc, vector in zip(documents, vectors, strict=True)
        }
        self._query_embeddings = query_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._query_embeddings.embed_query(text)


def load_vector_store(
    path: Path | None = None, embeddings: Embeddings | None = None
) -> InMemoryVectorStore:
    path = path if path is not None else INDEX_PATH
    if not path.exists():
        raise FileNotFoundError("知识库索引不存在，请先运行 uv run 08-build-index.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 同时支持之前示例产生的 camelCase JSON，无需重新调用文档 Embedding API。
    model = data.get("embedding_model") or data.get("embeddingModel")
    if not model:
        raise ValueError("索引缺少 Embedding 模型名称，请重新构建索引")
    documents = [
        Document(
            page_content=doc.get("page_content", doc.get("pageContent", "")),
            metadata=doc.get("metadata", {}),
        )
        for doc in data["documents"]
    ]
    vectors = data["vectors"]
    if not documents or len(documents) != len(vectors):
        raise ValueError("索引文档与向量数量不匹配，请重新构建索引")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("索引向量维度不一致，请重新构建索引")
    cached = CachedDocumentEmbeddings(
        documents,
        vectors,
        embeddings if embeddings is not None else embedding_model(model),
    )
    return InMemoryVectorStore.from_documents(documents, cached)
