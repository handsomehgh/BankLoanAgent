# author hgh
# version 2.0

"""
business knowledge collection init script (business_knowledge)

usage:
- create the business knowledge collection (schema v2)
- configure dense vector index(HNSW) and BM25 sparse vector index(SPARSE_INVERTED_INDEX)
- create inverted indexes for scalar filter fields

mode of operation:
    python pipelines/scripts/db_scripts/init_milvus_knowledge_col.py

schema v2（2026-08-08）:
- id 改为确定性编号（source_file:doc_seq:chunk_index），支持 upsert 与版本对比
- 移除图谱字段 entity_id/entity_type/relation_predicate
- 新增 question（FAQ专属）/ doc_version / ingest_batch_id（批次清理与回滚锚点）
- created_at/updated_at 由 VARCHAR ISO串 改为 INT64 Unix 秒时间戳

note: schema 不兼容旧表，变更前需先删除旧 collection 再重建
"""
import logging
import sys

from pymilvus import FieldSchema, DataType, Function, FunctionType, CollectionSchema, Collection, MilvusException
from pymilvus.orm import utility

from config.global_constant.constants import MemoryType, RegistryModules
from config.global_constant.fields import CommonFields
from config.models.retrieval_config import RetrievalConfig
from infra.database.collections_type import CollectionNames
from infra.database.milvus_client import MilvusClientManager
from utils.config_utils.get_config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE)

BM25_FUNCTION = Function(
    name="bm25_fn",
    function_type=FunctionType.BM25,
    input_field_names=[CommonFields.TEXT],
    output_field_names=["sparse_vector"]
)

# schema v2 字段定义，按职责分五组：主键与血缘 / 内容与向量 / 业务过滤 / 生命周期治理 / 扩展兜底
COL_FIELDS = [
    # ---- 主键与血缘 ----
    # 确定性id：{source_file}:{doc_seq}:{chunk_index}，如 faq.md:Q012:1
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=128),
    # 源文档稳定标识：{source_file}:{doc_seq}
    FieldSchema(name="parent_doc_id", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="chunk_index", dtype=DataType.INT64, default_value=0),
    # ---- 内容与向量 ----
    # text 同时是 BM25 Function 的输入，analyzer 变更需重建 collection
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"}
    ),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="term_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    # ---- 业务过滤 ----
    FieldSchema(name="product_type", dtype=DataType.VARCHAR, max_length=64, default_value="通用"),
    FieldSchema(name="topics", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=64),
    FieldSchema(name="regulation_names", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=8, max_length=64),
    # FAQ 原始问题文本，仅 faq 来源填充；512字节 ≈ 170个中文字
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512, nullable=True),
    # ---- 生命周期治理 ----
    FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32, default_value="active"),
    FieldSchema(name="confidence", dtype=DataType.FLOAT, default_value=0.0),
    # 源文档版本，对齐 md 文档头（如 v4.0）
    FieldSchema(name="doc_version", dtype=DataType.VARCHAR, max_length=16, default_value=""),
    # 导入批次号，全量重建后 delete("ingest_batch_id != 当前批次") 清理旧数据
    FieldSchema(name="ingest_batch_id", dtype=DataType.VARCHAR, max_length=32, default_value=""),
    # Unix 秒时间戳
    FieldSchema(name="created_at", dtype=DataType.INT64, default_value=0),
    FieldSchema(name="updated_at", dtype=DataType.INT64, default_value=0),
    # ---- 扩展兜底 ----
    # 低频字段（如 glossary 的 term/english）统一收进 extra，不开列
    FieldSchema(name="extra", dtype=DataType.JSON),
]

# 需要建倒排索引的标量过滤字段（与检索端 filter 表达式对齐）
SCALAR_INDEX_FIELDS = ["source_type", "product_type", "status", "source_file", "ingest_batch_id", "topics"]

# 向量索引计划：(字段名, 索引名, 索引参数来源)
VECTOR_INDEX_PLAN = [
    ("dense_vector", "dense_vector_idx", "dense"),
    ("term_vector", "term_vector_idx", "dense"),
    ("sparse_vector", "sparse_vector_idx", "sparse"),
]


def build_index_params(retrieval_config: RetrievalConfig) -> dict:
    """索引参数统一取自 retrieval_config.yaml，避免与配置双份维护漂移"""
    cfg = retrieval_config.index_params
    return {
        "dense": cfg["dense"].model_dump(),
        "sparse": cfg["sparse"].model_dump(),
    }


def create_collection_if_not_exist(name: str, fields: list, description: str = "") -> Collection:
    """create collection if not exists,and return it"""
    if utility.has_collection(collection_name=name):
        logger.info(f"Collection {name} already exists")
        col = Collection(name=name)
        col.load()
        return col

    logger.info(f"Creating collection {name}")
    schema = CollectionSchema(fields=fields, functions=[BM25_FUNCTION], description=description)
    try:
        col = Collection(name=name, schema=schema)
        logger.info(f"Collection {name} created successfully")
        return col
    except MilvusException as e:
        logger.error(f"Failed to create collection: {name} : {e}")
        raise


def create_index(collection: Collection, index_params: dict):
    """create vector indexes and scalar inverted indexes for collection"""
    # vector indexes
    for field_name, idx_name, param_key in VECTOR_INDEX_PLAN:
        if not collection.has_index(index_name=idx_name):
            collection.create_index(
                field_name=field_name,
                index_params=index_params[param_key],
                index_name=idx_name
            )
            utility.wait_for_index_building_complete(collection.name, idx_name)
            logger.info(f"Created index {idx_name} on {field_name}")
        else:
            logger.info(f"Index {idx_name} on {field_name} already exists")

    # scalar inverted indexes(high-frequency filter fields)
    for field in SCALAR_INDEX_FIELDS:
        idx_name = f"scalar_{field}_idx"
        if not collection.has_index(index_name=idx_name):
            try:
                logger.info(f"Creating scalar index {idx_name} on {collection.name}")
                collection.create_index(
                    field_name=field,
                    index_name=idx_name,
                    index_params={"index_type": "INVERTED"}
                )
                utility.wait_for_index_building_complete(collection.name, index_name=idx_name)
            except MilvusException as e:
                logger.error(f"Failed to create scalar index {idx_name} : {e}")


def load_collection(col: Collection):
    try:
        col.load()
        logger.info(f"Collection '{col.name}' loaded.")
    except MilvusException as e:
        logger.error(f"Failed to load collection '{col.name}': {e}")
        raise


# ========================= main process ==============================
def init_collections(retrieval_config: RetrievalConfig):
    index_params = build_index_params(retrieval_config)
    col = create_collection_if_not_exist(COLLECTION_NAME, COL_FIELDS, "business knowledge")
    create_index(col, index_params)
    load_collection(col)
    logger.info("Business knowledge collection initialized successfully.")


if __name__ == '__main__':
    try:
        registry = get_config()
        retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
        # establish the default milvus connection (same uri as retrieval/import)
        MilvusClientManager(retrieval_config.milvus_uri)
        init_collections(retrieval_config)
    except Exception:
        logger.exception("Initialization failed")
        sys.exit(1)
