"""轻量本地模型服务：用 sentence-transformers 加载 bge-m3 / bge-reranker-v2-m3，
暴露与 Xinference 兼容的 /v1/embeddings 和 /v1/rerank 接口，供 rag_bge 直接调用。

启动:
    .venv/bin/python server.py
默认监听 0.0.0.0:9997，模型从本地 models/ 目录加载。
"""

import os
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
EMB_PATH = os.getenv("EMB_MODEL_PATH", os.path.join(ROOT, "models", "bge-m3"))
RERANK_PATH = os.getenv("RERANK_MODEL_PATH", os.path.join(ROOT, "models", "bge-reranker-v2-m3"))
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "9997"))


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
print(f"[server] device={DEVICE}")
print(f"[server] loading embedding model: {EMB_PATH}")
embedder = SentenceTransformer(EMB_PATH, device=DEVICE)
print(f"[server] loading rerank model: {RERANK_PATH}")
reranker = CrossEncoder(RERANK_PATH, device=DEVICE)
print("[server] models loaded")

app = FastAPI(title="rag-bge local model server")


# ---- /v1/embeddings ----
class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    vectors = embedder.encode(
        req.input, normalize_embeddings=True, convert_to_numpy=True
    )
    data = [
        {"index": i, "object": "embedding", "embedding": vec.tolist()}
        for i, vec in enumerate(vectors)
    ]
    return {"object": "list", "model": req.model or "bge-m3", "data": data}


# ---- /v1/rerank ----
class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = False


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, doc] for doc in req.documents]
    scores = reranker.predict(pairs)
    ranked = sorted(
        ((i, float(s)) for i, s in enumerate(scores)),
        key=lambda x: x[1],
        reverse=True,
    )
    top_n = req.top_n or len(ranked)
    results = []
    for idx, score in ranked[:top_n]:
        item = {"index": idx, "relevance_score": score}
        if req.return_documents:
            item["document"] = {"text": req.documents[idx]}
        results.append(item)
    return {"model": req.model or "bge-reranker-v2-m3", "results": results}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
