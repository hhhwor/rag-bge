"""从 ModelScope 下载 bge-m3 / bge-reranker-v2-m3 到本地 models/ 目录。"""
import os
from modelscope import snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

targets = {
    "BAAI/bge-m3": os.path.join(MODELS_DIR, "bge-m3"),
    "BAAI/bge-reranker-v2-m3": os.path.join(MODELS_DIR, "bge-reranker-v2-m3"),
}

for repo, local_dir in targets.items():
    print(f"[download] {repo} -> {local_dir}", flush=True)
    path = snapshot_download(repo, local_dir=local_dir)
    print(f"[done] {repo} at {path}", flush=True)

print("ALL_DOWNLOADS_DONE", flush=True)
