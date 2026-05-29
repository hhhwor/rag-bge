"""查询流程：bge-m3 向量召回 Top-K -> bge-reranker-v2-m3 精排 Top-N。

用法:
    python -m rag_bge.retriever "你的问题"
"""

import sys
from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import config
from .models import BGEEmbeddings, BGEReranker


@dataclass
class Hit:
    content: str
    score: float          # rerank 相关性分数
    metadata: dict


class Searcher:
    def __init__(self):
        self.embeddings = BGEEmbeddings()
        self.reranker = BGEReranker()
        # allow_dangerous_deserialization：FAISS 本地索引含 pickle，仅加载自己生成的索引
        self.store = FAISS.load_local(
            config.INDEX_DIR,
            self.embeddings,
            index_name=config.INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

    def search(
        self,
        query: str,
        recall_k: int = config.RECALL_TOP_K,
        rerank_n: int = config.RERANK_TOP_N,
    ) -> List[Hit]:
        # 1) 向量召回
        candidates: List[Document] = self.store.similarity_search(query, k=recall_k)
        if not candidates:
            return []

        # 2) rerank 精排
        texts = [d.page_content for d in candidates]
        ranked = self.reranker.rerank(query, texts, top_n=rerank_n)

        return [
            Hit(
                content=candidates[idx].page_content,
                score=score,
                metadata=candidates[idx].metadata,
            )
            for idx, score in ranked
        ]


def main():
    if len(sys.argv) < 2:
        print('用法: python -m rag_bge.retriever "你的问题"')
        return
    query = sys.argv[1]
    hits = Searcher().search(query)
    print(f"查询: {query}\n共 {len(hits)} 条结果:\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h.score:.4f}  source={h.metadata.get('source', 'N/A')}")
        print(f"    {h.content[:200].strip()}\n")


if __name__ == "__main__":
    main()
