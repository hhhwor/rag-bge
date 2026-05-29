"""模型封装：通过 HTTP 调用 Xinference 上的 bge-m3 / bge-reranker-v2-m3。

- BGEEmbeddings 实现 LangChain 的 Embeddings 接口，可直接喂给 FAISS。
- BGEReranker 调用 Xinference 的 /v1/rerank 端点做精排。
两者都不在本进程加载权重，只发 HTTP 请求。
"""

from typing import List, Tuple

import requests
from langchain_core.embeddings import Embeddings

from . import config


class BGEEmbeddings(Embeddings):
    """调用 Xinference 的 OpenAI 兼容 /v1/embeddings 接口。"""

    def __init__(
        self,
        base_url: str = config.XINFERENCE_BASE_URL,
        model_uid: str = config.EMBEDDING_MODEL_UID,
        timeout: int = 60,
    ):
        self.endpoint = f"{base_url.rstrip('/')}/v1/embeddings"
        self.model_uid = model_uid
        self.timeout = timeout

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            self.endpoint,
            json={"model": self.model_uid, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # 按 index 排序，保证与输入顺序一致
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 分批发送，避免单请求过大
        batch_size = 32
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[i : i + batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class BGEReranker:
    """调用 Xinference 的 /v1/rerank 接口，对候选文档精排。"""

    def __init__(
        self,
        base_url: str = config.XINFERENCE_BASE_URL,
        model_uid: str = config.RERANK_MODEL_UID,
        timeout: int = 60,
    ):
        self.endpoint = f"{base_url.rstrip('/')}/v1/rerank"
        self.model_uid = model_uid
        self.timeout = timeout

    def rerank(
        self, query: str, documents: List[str], top_n: int = config.RERANK_TOP_N
    ) -> List[Tuple[int, float]]:
        """返回 [(原始下标, 相关性分数), ...]，按分数从高到低，长度 <= top_n。"""
        if not documents:
            return []
        resp = requests.post(
            self.endpoint,
            json={
                "model": self.model_uid,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return [(r["index"], r["relevance_score"]) for r in results]
