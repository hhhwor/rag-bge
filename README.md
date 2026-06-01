# rag-bge

基于 **LangChain + FAISS** 的向量索引 pipeline，模型私有化部署用 **bge-m3**（embedding）+ **bge-reranker-v2-m3**（rerank），通过 **Xinference** 提供 HTTP 服务。独立于 RAGFlow，但模型调用方式与 RAGFlow 生态一致，便于后续打通。

## 流程

```
文档加载 → 切分 → bge-m3 向量化 → FAISS 索引（建库）
查询 → bge-m3 向量化 → FAISS 召回 Top-K → bge-reranker-v2-m3 精排 Top-N → 结果
```

## 1. 部署模型（Xinference）

```bash
# 安装（建议单独的环境 / 单独的 GPU 机器）
pip install "xinference[all]"

# 启动服务（默认端口 9997）
xinference-local --host 0.0.0.0 --port 9997
```

启动后，注册两个模型（命令行或 Web UI http://127.0.0.1:9997）：

```bash
# embedding 模型，--model-uid 要和 config.EMBEDDING_MODEL_UID 一致
xinference launch --model-name bge-m3 --model-type embedding --model-uid bge-m3

# rerank 模型
xinference launch --model-name bge-reranker-v2-m3 --model-type rerank --model-uid bge-reranker-v2-m3
```

> 私有化离线环境：先把模型权重放到本地，用 `XINFERENCE_MODEL_SRC` 或 Web UI 指定本地路径加载，无需联网下载。

验证服务可用：

```bash
curl http://127.0.0.1:9997/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":["测试"]}'
```

## 2. 安装本项目依赖

```bash
cd ~/workspace/rag-bge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 配置

默认连 `http://127.0.0.1:9997`。若 Xinference 在别的机器/端口，用环境变量覆盖（见 `rag_bge/config.py`）：

```bash
export XINFERENCE_BASE_URL=http://10.0.0.5:9997
export EMBEDDING_MODEL_UID=bge-m3
export RERANK_MODEL_UID=bge-reranker-v2-m3
```

## 4. 建索引

把文档（`.txt` / `.md` / `.pdf`）放进 `data/`，然后：

```bash
python -m rag_bge.indexer
# 或指定目录： python -m rag_bge.indexer /path/to/docs
```

索引保存在 `index/faiss_index.*`。

## 5. 查询

```bash
python -m rag_bge.retriever "博士后出站需要什么"
```

也可在代码里调用：

```python
from rag_bge.retriever import Searcher

s = Searcher()
hits = s.search("博士后出站需要什么", recall_k=20, rerank_n=5)
for h in hits:
    print(h.score, h.metadata, h.content[:80])
```

## 6. 小数据验证（跑通 pipeline）

用 `data/` 下的示例文档（博士后管理、职工请假、差旅报销三篇不同主题）端到端验证一遍。

```bash
# 0) 启动本地模型服务（不走 Xinference 时用这个，从 models/ 加载权重）
.venv/bin/python server.py            # 监听 0.0.0.0:9997，默认 CPU，有 GPU 自动用 cuda
curl http://127.0.0.1:9997/health     # {"status":"ok","device":"cpu"} 即就绪

# 1) 建索引：3 篇文档 -> 3 个 chunk -> FAISS 落盘到 index/
.venv/bin/python -m rag_bge.indexer

# 2) 查询：召回 Top-K -> reranker 精排 Top-N
.venv/bin/python -m rag_bge.retriever "博士后出站需要什么"
```

跨主题查询应能把对应文档排到第一，干扰项分数被明显拉低，说明召回 + 精排链路正常：

| 查询 | Top-1 命中 | rerank 分数 | 次高分（干扰项） |
|---|---|---|---|
| 博士后出站需要什么 | sample_postdoc.md | ~0.97 | ~0.02 |
| 工作满15年有几天年假 | sample_leave.md | ~0.66 | ~0.003 |
| 出差住宿费怎么报销 | sample_reimburse.md | ~0.90 | ~0.005 |

> 示例文档较短，每篇约切成 1 个 chunk；换真实语料时按文档粒度调 `CHUNK_SIZE`（默认 512）。

## 模块说明

| 文件 | 作用 |
|---|---|
| `rag_bge/config.py` | 集中配置（服务地址、切分参数、Top-K/N、路径） |
| `rag_bge/models.py` | `BGEEmbeddings`（LangChain Embeddings 接口）/ `BGEReranker`，均走 HTTP |
| `rag_bge/indexer.py` | 加载 → 切分 → 向量化 → FAISS 落盘 |
| `rag_bge/retriever.py` | 召回 + rerank 精排 |

## 后续可扩展

- 换生产级向量库：把 `FAISS` 换成 LangChain 的 `Milvus` / `Qdrant` / `PGVector`，`models.py` 不动。
- 与 RAGFlow 打通：RAGFlow 同样用 Xinference 调这两个模型，可共用同一套模型服务。
- 接入 LLM 生成答案：在 `retriever` 之后加一步，把 Top-N chunk 拼进 prompt 调 LLM。
