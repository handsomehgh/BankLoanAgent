import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from pymilvus import (
    MilvusException,
)

from config.global_constant.constants import RegistryModules, MemoryType, KnowledgeFileSourceType
from config.global_constant.fields import CommonFields
from config.prompt_hub import PromptHub
from config.registry import ConfigRegistry
from infra.database.collections_type import CollectionNames
from infra.database.milvus_client import MilvusClientManager
from modules.module_services.chat_models import RobustLLM
from modules.module_services.embeddings import RobustEmbeddings, RobustLocalEmbeder
from pipelines.constant import FileMetadata
from utils.config_utils.get_config import get_config
from utils.faq_similar_generator import FaqSimilarGenerator
from utils.summary_know_generator import SummaryKnowledgeGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# metadata 中已映射到独立列的键，其余键统一兑底收进 extra
MAPPED_META_KEYS = {
    FileMetadata.CHUNK_ID, FileMetadata.SOURCE_TYPE, FileMetadata.PRODUCT_TYPE,
    FileMetadata.SOURCE_FILE, FileMetadata.PARENT_DOC_ID, FileMetadata.CHUNK_INDEX,
    FileMetadata.CONFIDENCE, FileMetadata.TOPICS, FileMetadata.REGULATION_NAMES,
    FileMetadata.QUESTION, FileMetadata.DOC_VERSION,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGIndexer:
    def __init__(self,registry: ConfigRegistry,embedder:RobustLocalEmbeder,llm_client: RobustLLM):
        self.registry = registry
        self.embedder = embedder

        self.retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
        self.client = MilvusClientManager(self.retrieval_config.milvus_uri)
        self.llm_client = llm_client
        self.collection = self.client.get_collection(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE))

        multi_cfg = self.retrieval_config.multi_vector
        self.summary_generator = None
        if multi_cfg.enable and multi_cfg.summary_vector:
            self.summary_generator = SummaryKnowledgeGenerator(self.retrieval_config,llm_client)
            logger.info("Summary generator is enabled，source type：%s，minimum len：%d",
                        multi_cfg.summary_config.enabled_sources, multi_cfg.summary_config.min_chunk_length)

        self.faq_similar_generator = None
        if multi_cfg.enable and multi_cfg.faq_similar_vector:
            faq_config = getattr(multi_cfg.faq_similar_config, 'faq_similar_config', {})
            self.faq_similar_generator = FaqSimilarGenerator(faq_config,llm_client, PromptHub(registry))
            logger.info("FAQ Similar Question Generator is enabled")

    def load_chunks(self) -> List[Dict[str, Any]]:
        """load chunk for phase two"""
        file_config = self.registry.get_config(RegistryModules.FILE_PROCESS)
        input_file = PROJECT_ROOT / file_config.stage2_output_dir / "chunked_docs.jsonl"

        if not input_file.exists():
            raise FileNotFoundError(f"Phase two output file does not exist: {input_file}")

        chunks = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        logger.info(f"load {len(chunks)} chunks")
        return chunks

    def _generate_term_text(self, metadata: Dict[str, Any]) -> str:
        parts = []
        topics = metadata.get(FileMetadata.TOPICS, [])
        if isinstance(topics, list):
            parts.extend(topics)
        elif isinstance(topics, str):
            parts.append(topics)
        regs = metadata.get(FileMetadata.REGULATION_NAMES, [])
        if isinstance(regs, list):
            parts.extend(regs)
        elif isinstance(regs, str):
            parts.append(regs)
        return " ".join(parts)

    def _generate_summary(self, batch: List[Dict[str, Any]], dense_vectors: List[List[float]]) -> List[List[float]]:
        if self.summary_generator:
            summary_texts = []
            summary_indices = []
            for idx, item in enumerate(batch):
                content = item[CommonFields.CONTENT]
                source = item[CommonFields.METADATA].get(FileMetadata.SOURCE_TYPE, "")
                length = len(content)
                if self.summary_generator.should_generate(source, length):
                    summary_texts.append(content)
                    summary_indices.append(idx)

            generated = []
            for text in summary_texts:
                result = self.summary_generator.generate_summary(text)
                if result:
                    generated.append(result)
                else:
                    generated.append(text)

            if generated:
                summary_vectors_list = self.embedder.embed_documents(generated)
                summary_vectors = list(dense_vectors)
                for i, idx in enumerate(summary_indices):
                    summary_vectors[idx] = summary_vectors_list[i]
            else:
                summary_vectors = dense_vectors
        else:
            summary_vectors = dense_vectors

        return summary_vectors

    def _generate_similar_faq(self, batch: List[Dict[str, Any]], dense_vectors: List[List[float]]) -> List[List[float]]:
        if self.faq_similar_generator:
            faq_texts = []
            faq_indices = []
            for idx, item in enumerate(batch):
                source = item[CommonFields.METADATA].get(FileMetadata.SOURCE_TYPE, "")
                if source != KnowledgeFileSourceType.FAQ:
                    continue
                content = item[CommonFields.CONTENT]
                question = content
                if "Q" in content and "A:" in content:
                    q_start = content.find("Q:") + 2
                    a_start = content.find("A:")
                    if q_start < a_start:
                        question = content[q_start:a_start].strip()
                faq_texts.append(question)
                faq_indices.append(idx)

            if faq_texts:
                generated = []
                for q in faq_texts:
                    result = self.faq_similar_generator.generate_faq(q)
                    if result:
                        generated.append(result)
                    else:
                        generated.append(q)
                faq_vectors_list = self.embedder.embed_documents(generated)
                faq_vectors = list(dense_vectors)
                for i, idx in enumerate(faq_indices):
                    faq_vectors[idx] = faq_vectors_list[i]
            else:
                faq_vectors = dense_vectors
        else:
            faq_vectors = dense_vectors
        return faq_vectors

    def _ensure_list(self, val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            # 可能是逗号分隔，拆分为列表
            return [item.strip() for item in val.split(",") if item.strip()]
        return []

    @staticmethod
    def _truncate_utf8(text: str, max_bytes: int) -> str:
        """按 UTF-8 字节数截断（Milvus VARCHAR max_length 是字节语义）"""
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def index(self):
        chunks = self.load_chunks()
        if not chunks:
            return

        # 导入批次号：全量重建后据此清理旧批次数据，支持回滚与审计
        doc_version = chunks[0][CommonFields.METADATA].get(FileMetadata.DOC_VERSION, "")
        batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{doc_version}".rstrip("-")
        logger.info(f"Ingest batch id：{batch_id}")

        retrieval_config = self.registry.get_config(RegistryModules.RETRIEVAL)
        batch_size = retrieval_config.insert_batch_size

        total = len(chunks)
        logger.info(f"Starting index {total} chunk，batch size：{batch_size}")

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            try:
                entities = self._prepare_batch(batch, batch_id)
                self.collection.upsert(entities)
                logger.info(f"Upserted {min(start + batch_size, total)}/{total}")
            except MilvusException as e:
                logger.error(f"Upsert failed (batch {start // batch_size}): {e}")
                raise

        self.collection.flush()

        # 清理非当前批次的历史数据（仅全量导入语义下安全）
        try:
            self.collection.delete(f'ingest_batch_id != "{batch_id}"')
            self.collection.flush()
            logger.info(f"Stale batches cleaned，current batch：{batch_id}")
        except MilvusException as e:
            logger.error(f"Failed to clean stale batches: {e}")

        logger.info(f"import success，total number of entities in the set：{self.collection.num_entities}")

    def _prepare_batch(self, batch: List[Dict[str, Any]], batch_id: str) -> List[Dict]:
        texts = [item[CommonFields.CONTENT] for item in batch]
        metadatas = [item[CommonFields.METADATA] for item in batch]

        dense_vectors = self.embedder.embed_documents(texts)

        retrieval_config = self.registry.get_config(RegistryModules.RETRIEVAL)

        term_vectors = dense_vectors
        faq_vectors = dense_vectors
        summary_vectors = dense_vectors
        if retrieval_config.multi_vector.enable:
            if retrieval_config.multi_vector.term_vector:
                term_texts = [self._generate_term_text(m) for m in metadatas]
                term_vectors = self.embedder.embed_documents(term_texts)

            if retrieval_config.multi_vector.faq_similar_vector:
                faq_vectors = self._generate_similar_faq(batch,dense_vectors)

            if retrieval_config.multi_vector.summary_vector:
                summary_vectors = self._generate_summary(batch, dense_vectors)

        entities = []
        now_ts = int(time.time())
        for i in range(len(batch)):
            meta = metadatas[i]

            # 未映射到独立列的元数据（如 glossary 的 term/english、faq 的 answer）兑底收进 extra
            extra = {k: v for k, v in meta.items() if k not in MAPPED_META_KEYS}
            answer = meta.get(FileMetadata.ANSWER)
            if answer:
                extra["answer"] = answer

            question = meta.get(FileMetadata.QUESTION) or None
            if question:
                question = self._truncate_utf8(question, 512)

            entity = {
                "id": meta.get(FileMetadata.CHUNK_ID, ""),
                "text": texts[i],
                "source_type": meta.get(FileMetadata.SOURCE_TYPE, ""),
                "product_type": meta.get(FileMetadata.PRODUCT_TYPE, "通用"),
                "source_file": meta.get(FileMetadata.SOURCE_FILE, ""),
                "parent_doc_id": meta.get(FileMetadata.PARENT_DOC_ID, ""),
                "chunk_index": meta.get(FileMetadata.CHUNK_INDEX, 0),
                "question": question,
                "status": "active",
                "confidence": meta.get(FileMetadata.CONFIDENCE, 0.7),
                "doc_version": meta.get(FileMetadata.DOC_VERSION, ""),
                "ingest_batch_id": batch_id,
                "created_at": now_ts,
                "updated_at": now_ts,
                "topics": self._ensure_list(meta.get(FileMetadata.TOPICS, [])),
                "regulation_names": self._ensure_list(meta.get(FileMetadata.REGULATION_NAMES, [])),
                "extra": extra,
                "dense_vector": dense_vectors[i]
            }

            if term_vectors is not None:
                entity["term_vector"] = term_vectors[i]
            entities.append(entity)

        return entities


if __name__ == "__main__":
    registry = get_config()
    llm_config = registry.get_config(RegistryModules.LLM)
    embedder = RobustLocalEmbeder(
        base_url="http://127.0.0.1:8001/v1",
        model_name="no_need"
    )

    precise_llm = RobustLLM(
        temperature=llm_config.precise_temperature,
        api_key=llm_config.deepseek_api_key,
        base_url=llm_config.deepseek_base_url,
        model=llm_config.deepseek_llm_name,
        provider=llm_config.openai_provider
    )
    indexer = RAGIndexer(registry=registry, embedder=embedder,llm_client=precise_llm)
    indexer.index()