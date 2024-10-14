from loguru import logger
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch
import os
from dotenv import load_dotenv

BATCH_SIZE = 100
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 加载.env文件
load_dotenv(override=True)

NOVEL_FILE_DIR = os.getenv("NOVEL_FILE_DIR")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese"),
    embed_batch_size=BATCH_SIZE,
    device=device,
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "lerry"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "novels"),
}

print(DB_CONFIG)

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
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def create_and_insert_document(index, chapter_file):
    try:
        with open(chapter_file, "r", encoding="utf-8") as f:
            content = f.read()

        doc_id = os.path.basename(chapter_file)
        title = doc_id.split("_")[1] if "_" in doc_id else doc_id

        doc = Document(
            doc_id=doc_id,
            text=content,
            extra_info={
                "title": title,
                "chapter": doc_id,
            },
        )
        index.insert(doc)
        return True
    except Exception as e:
        logger.error(f"创建或插入文档时发生错误: {str(e)}")
        return False


def embed_novels():
    try:
        logger.info("开始小说嵌入过程")

        logger.info("正在初始化向量存储...")
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        logger.info("成功初始化向量存储")

        total_synced = 0
        chapter_files = sorted(
            [f for f in os.listdir(NOVEL_FILE_DIR) if f.endswith(".txt")]
        )

        for chapter_file in chapter_files:
            full_path = os.path.join(NOVEL_FILE_DIR, chapter_file)

            logger.info(f"正在处理章节文件: {chapter_file}")
            result = create_and_insert_document(index, full_path)

            if result:
                total_synced += 1
                logger.info(f"成功添加章节到向量存储")

            logger.info(f"已同步 {total_synced} 个章节。")

        logger.info(f"小说嵌入完成���共同步 {total_synced} 个章节")
    except Exception as e:
        logger.error(f"同步过程中发生错误: {str(e)}")
        logger.exception("错误详情:")


if __name__ == "__main__":
    embed_novels()
