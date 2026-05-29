"""建索引流程：加载文档 -> 切分 -> bge-m3 embedding -> FAISS 落盘。

用法:
    python -m rag_bge.indexer            # 索引 data/ 目录下所有 .txt/.md/.pdf
    python -m rag_bge.indexer ./somedir  # 索引指定目录
"""

import os
import sys
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .models import BGEEmbeddings


def load_documents(path: str) -> List[Document]:
    """加载目录下的 .txt / .md / .pdf。简单实现，按需扩展 loader。"""
    docs: List[Document] = []
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            try:
                if ext in ("txt", "md"):
                    with open(fpath, encoding="utf-8") as f:
                        text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": fpath}))
                elif ext == "pdf":
                    from langchain_community.document_loaders import PyPDFLoader

                    docs.extend(PyPDFLoader(fpath).load())
                else:
                    continue
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {fpath}: {e}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(data_dir: str = None) -> None:
    data_dir = data_dir or config.DATA_DIR
    print(f"[1/4] 加载文档: {data_dir}")
    raw = load_documents(data_dir)
    if not raw:
        print("没有找到可索引的文档（支持 .txt/.md/.pdf），请放入 data/ 目录。")
        return
    print(f"      共 {len(raw)} 个文档")

    print("[2/4] 切分 chunk")
    chunks = split_documents(raw)
    print(f"      共 {len(chunks)} 个 chunk")

    print("[3/4] 调用 bge-m3 生成向量并构建 FAISS 索引")
    embeddings = BGEEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)

    print(f"[4/4] 保存索引到 {config.INDEX_DIR}/{config.INDEX_NAME}")
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    store.save_local(config.INDEX_DIR, index_name=config.INDEX_NAME)
    print("完成。")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    build_index(target)
