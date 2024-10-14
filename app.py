from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch
import os
import dotenv

dotenv.load_dotenv(override=True)

app = FastAPI()


# 设置嵌入模型
BATCH_SIZE = 32  # 您可以根据需要调整这个值
device = "cuda" if torch.cuda.is_available() else "cpu"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="shibing624/text2vec-base-chinese",
    embed_batch_size=BATCH_SIZE,
    device=device,
)

# 设置LLM模型
llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1024,
    system_prompt="你是一个小说阅读AI助手，请用中文回答所有问题。",
)

# 数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "lerry"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "novels"),
}

# 初始化向量存储
vector_store = PGVectorStore.from_params(
    **DB_CONFIG,
    table_name="novels_vectors",
    embed_dim=768,
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

# 创建存储上下文和索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

# 创建查询引擎
query_engine = index.as_query_engine(llm=llm)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 使用查询引擎获取响应
        response = query_engine.query(request.message)

        return ChatResponse(response=str(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天过程中发生错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
