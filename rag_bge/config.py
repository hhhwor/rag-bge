"""全局配置。所有可调参数集中在此，通过环境变量覆盖。"""

import os

# ---- Xinference 服务地址 ----
# 启动 Xinference 后，模型通过这个 base_url 访问（OpenAI 兼容接口在 /v1）
XINFERENCE_BASE_URL = os.getenv("XINFERENCE_BASE_URL", "http://127.0.0.1:9997")

# 在 Xinference 里 launch 模型时指定的 model_uid（不是模型名，是实例 uid）
EMBEDDING_MODEL_UID = os.getenv("EMBEDDING_MODEL_UID", "bge-m3")
RERANK_MODEL_UID = os.getenv("RERANK_MODEL_UID", "bge-reranker-v2-m3")

# ---- 切分参数 ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# ---- 检索参数 ----
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "20"))   # 向量召回数量
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))    # 重排后保留数量

# ---- 路径 ----
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_ROOT, "data"))
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(_ROOT, "index"))
INDEX_NAME = os.getenv("INDEX_NAME", "faiss_index")
