# author hgh
# version 1.0
import json
import logging
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.global_constant.constants import RegistryModules
from config.global_constant.fields import CommonFields
from config.models.file_process_config import FileProcessConfig
from config.registry import ConfigRegistry
from pipelines.constant import FileMetadata
from utils.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


class IntelligentChunk:
    def __init__(self, config: FileProcessConfig):
        self.config = config
        self.chunk_config = self.config.chunking

    def _split_by_structure(self, text: str, delimiter: str) -> List[str]:
        parts = re.split(f'({re.escape(delimiter)})', text)
        chunks, current = [], ""
        for part in parts:
            if part == delimiter:
                if current.strip():
                    chunks.append(current.strip())
                current = part
            else:
                current += part
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text]

    def chunk_document(self, docs: List[Document]) -> List[Document]:
        final_chunks = []
        strategies = self.chunk_config.strategies
        default_strategy = self.chunk_config.default
        confidence_map = self.config.metadata_extraction.default_confidence_by_source
        fallback_conf = self.config.metadata_extraction.fallback_confidence

        discard_samples = []
        for doc in docs:
            source = doc.metadata.get(FileMetadata.SOURCE_TYPE, "unknown")
            strategy = strategies.get(source, default_strategy)
            logger.debug(f"Process source={source}, method={strategy.method}")

            if strategy.method == "no_split":
                chunks = [doc.page_content]
            elif strategy.method == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=strategy.chunk_size,
                    chunk_overlap=strategy.chunk_overlap,
                    separators=self.chunk_config.separators,
                )
                chunks = splitter.split_text(doc.page_content)
            elif strategy.method == "structure_then_recursive":
                raw_chunks = self._split_by_structure(doc.page_content, strategy.structure_delimiter)
                chunks = []
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=strategy.chunk_size,
                    chunk_overlap=strategy.chunk_overlap,
                    separators=self.chunk_config.separators,
                )

                for rc in raw_chunks:
                    if len(rc) <= strategy.chunk_size:
                        chunks.append(rc)
                    else:
                        chunks.extend(splitter.split_text(rc))
            else:
                logger.warning(f"Unknow chunk method {strategy.method}，use default strategy")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=default_strategy.chunk_size,
                    chunk_overlap=default_strategy.chunk_overlap,
                    separators=self.chunk_config.separators,
                )
                chunks = splitter.split_text(doc.page_content)

            strategy_min_len = strategy.min_chunk_length if strategy.min_chunk_length is not None else self.chunk_config.min_chunk_length

            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < strategy_min_len:
                    discard_samples.append(chunk)
                    continue
                new_meta = dict(doc.metadata)
                new_meta[FileMetadata.PARENT_DOC_ID] = doc.metadata.get(FileMetadata.CHUNK_ID, "")
                new_meta[FileMetadata.CONFIDENCE] = confidence_map.get(source, fallback_conf)
                new_meta[FileMetadata.CHUNK_ID] = str(uuid.uuid4())

                new_doc = Document(page_content=chunk, metadata=new_meta)
                final_chunks.append(new_doc)

            parent_groups = defaultdict(list)
            for chunk in final_chunks:
                parent_id = chunk.metadata.get(FileMetadata.PARENT_DOC_ID, "")
                parent_groups[parent_id].append(chunk)

            for parent_id, chunks in parent_groups.items():
                for idx, chunk in enumerate(chunks):
                    chunk.metadata[FileMetadata.CHUNK_INDEX] = idx

        groups = defaultdict(list)
        for chunk in final_chunks:
            parent_id = chunk.metadata.get(FileMetadata.PARENT_DOC_ID, "unknown")
            groups[parent_id].append(chunk)

        for parent_id, chunks in groups.items():
            for idx, chunk in enumerate(chunks):
                chunk.metadata[FileMetadata.CHUNK_INDEX] = idx + 1

        if discard_samples:
            logger.info(f"丢弃了 {len(discard_samples)} 个短块，前3个样本：")
            for d in discard_samples[:3]:
                logger.info(f"  [{len(d)} chars] {d[:100]}")

        logger.info(f"分块完成：{len(docs)} 个输入文档 → {len(final_chunks)} 个 chunk")
        return final_chunks


def run_chunking():
    registry = ConfigRegistry()
    registry.register_model(
        RegistryModules.FILE_PROCESS, FileProcessConfig,
        Path(PROJECT_ROOT / "config" / "rules" / "file_process_config.yaml")
    )
    registry.load_all()

    file_config = registry.get_config(RegistryModules.FILE_PROCESS)

    input_file = PROJECT_ROOT / file_config.stage1_output_dir / "preprocessed_docs.jsonl"
    if not input_file.exists():
        logger.error(f"第一阶段输出文件不存在: {input_file}")
        return

    documents = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc = Document(page_content=obj[CommonFields.CONTENT], metadata=obj[CommonFields.METADATA])
            documents.append(doc)

    chunker = IntelligentChunk(file_config)
    chunks = chunker.chunk_document(documents)

    output_file = PROJECT_ROOT / file_config.stage2_output_dir / "chunked_docs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps({
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }, ensure_ascii=False) + '\n')
    logger.info(f"分块结果已保存至 {output_file}")


if __name__ == '__main__':
    run_chunking()
