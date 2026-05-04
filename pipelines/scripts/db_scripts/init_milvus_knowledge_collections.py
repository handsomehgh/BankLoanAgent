# author hgh
# version 1.0
import logging
import sys
from pathlib import Path

from pymilvus import FieldSchema, DataType, connections, Collection, Function, FunctionType, CollectionSchema, \
    MilvusException
from pymilvus.orm import utility

from config.global_constant.constants import MemoryType
from infra.collections import CollectionNames
from pipelines.scripts.db_scripts.init_milvus_memory_collecitons import BM25_FUNCTION

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COL_FIELDS = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                analyzer_params={"type": "chinese"}),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="last_updated", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=32, default_value=""),
    FieldSchema(name="file_url", dtype=DataType.VARCHAR, max_length=1024, default_value=""),
    FieldSchema(name="external_id", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20,
                max_length=20),
    FieldSchema(name="extra", dtype=DataType.JSON),
    FieldSchema(name="data_source_type", dtype=DataType.VARCHAR, max_length=32, default_value=""),
    FieldSchema(name="collected_at", dtype=DataType.VARCHAR, max_length=32, default_value="")
]

DENSE_INDEX_PARAM = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}

SPARSE_INDEX_PARAM = {
    "metric_type": "BM25",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {"drop_ratio_build": 0.2}
}

BM25_FUNCTION = Function(
    name="bm25_fn",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse_vector"]
)
SCALAR_INDEX_FIELDS = ["category", "last_updated", "version", "source"]


def connect_milvus():
    try:
        connections.connect(
            alias="default",
            uri="http://192.168.24.128:19530",
            timeout=30
        )
        logger.info(f"Connected to milvus")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        sys.exit(1)


def create_collection_if_not_exist(name: str, fields: list, description: str = ""):
    if utility.has_collection(collection_name=name):
        logger.info(f"collection {name} already exists")
        col = Collection(name=name)
        col.load()
        return col

    logger.info(f"Creating collection {name}")
    schema = CollectionSchema(fields=fields, functions=[BM25_FUNCTION])
    try:
        col = Collection(name=name, schema=schema)
        logger.info(f"Collection {name} created successfully")
        return col
    except MilvusException as e:
        logger.error(f"Failed to create collection: {name} : {e}")
        raise


def create_index(collection: Collection, dense_field: str = "dense_vector", sparse_field: str = "sparse_vector"):
    dense_idx_name = f"{dense_field}_idx"
    if not collection.has_index(index_name=dense_idx_name):
        collection.create_index(
            index_name=dense_idx_name,
            index_params=DENSE_INDEX_PARAM,
            field_name=dense_field
        )
        utility.wait_for_index_building_complete(collection.name, dense_idx_name)
    else:
        logger.info(f"Dense index on {dense_field} already exists")

    sparse_idx_name = f"{sparse_field}_idx"
    if not collection.has_index(index_name=sparse_idx_name):
        collection.create_index(
            field_name=sparse_field,
            index_params=SPARSE_INDEX_PARAM,
            index_name=sparse_idx_name
        )
        utility.wait_for_index_building_complete(collection.name, sparse_idx_name)
    else:
        logger.info(f"Sparse index on {sparse_field} already exists")

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


def load_collection(collection: Collection):
    try:
        collection.load()
        logger.info(f"Collection '{collection.name}' loaded.")
    except Exception as e:
        logger.error(f"Failed to load collection '{collection.name}': {e}")
        raise


def init_collections():
    connect_milvus()

    col = create_collection_if_not_exist(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE), COL_FIELDS,
                                         "knowledge collection")
    create_index(col)
    load_collection(col)
    logger.info(f"Collection '{col.name}' created successfully")


if __name__ == '__main__':
    try:
        init_collections()
    except MilvusException as e:
        logger.error(f"Initialization failed")
        sys.exit(1)
