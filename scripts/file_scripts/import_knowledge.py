import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config.constants import MemoryType
from config.settings import agentConfig
from memory.memory_vector_store.milvus_vector_store import MilvusVectorStore
from memory.models.memory_data.memory_schema import BusinessKnowledge

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
MAX_CHUNK_SIZE = 800  # maximum character number of single chunk
OVERLAP_SIZE = 50  # number of overlapping characters when splitting an extra-long block
TABLE_PIPE_RATIO = 0.3

RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "；", "？", "！", " ", ""]
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


# ======================= hybrid blocker ======================
def chunk_document(filepath: Path) -> List[Dict[str, str]]:
    """
    对单个 Markdown 文件分块，自动识别表格 / 标题型文档。
    返回列表: [{"title": ..., "content": ...}, ...]
    """
    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return []

    if _is_table_document(raw_text):
        logger.info(f"Detected as table document: {filepath.name}")
        return _parse_table_document(raw_text)

    logger.info(f"Detected as heading document: {filepath.name}")
    return _parse_heading_document(raw_text)


def _parse_table_document(content: str) -> List[Dict[str, str]]:
    chunks = []
    lines = [line for line in content.split("\n") if line.strip().startswith("|")]
    data_lines = [l for l in lines if not re.match(r'^\|[\s\-:\|]+$', l)]
    if len(data_lines) < 2:
        return chunks

    header = [c.strip() for c in data_lines[0].split("|")[1:-1]]
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != len(header):
            continue
        if len(header) >= 3:
            term, eng, expl = cells[0], cells[1], cells[2]
            content = f"{term} ({eng}): {expl}"
        else:
            content = " ".join([f"{h}: {c}" for h, c in zip(header, cells)])
        chunks.append({
            "title": f"术语: {cells[0]}",
            "content": content,
        })
    return chunks


def _parse_heading_document(content: str) -> List[Dict[str, str]]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False
    )

    try:
        md_docs = splitter.split_text(content)
    except Exception as e:
        logger.error(f"MarkdownHeaderTextSplitter failed: {e}")
        return []

    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        length_function=len,
        separators=RECURSIVE_SEPARATORS,
    )

    final_chunks: List[Dict[str, str]] = []
    for doc in md_docs:
        title = _extract_title_from_metadate(doc.metadata)
        content = doc.page_content.strip()
        if not content:
            continue

        if len(content) <= MAX_CHUNK_SIZE:
            final_chunks.append({
                "title": title,
                "content": f"{title}\n{content}",
            })
            continue

        sub_texts = length_splitter.split_text(content)
        for i, sub in enumerate(sub_texts):
            sub_title = f"{title} (段{i + 1})" if len(sub_texts) > 1 else title
            final_chunks.append({
                "title": sub_title,
                "content": f"{sub_title}\n{sub}",
            })
    return final_chunks


def _is_table_document(raw_text: str) -> bool:
    lines = raw_text.split("\n")
    if not lines:
        return False
    pipe_count = sum(1 for line in lines if line.strip().startswith("|"))
    return (pipe_count / len(lines)) > TABLE_PIPE_RATIO


def _extract_title_from_metadate(metadata: dict) -> str:
    for key in reversed(HEADERS_TO_SPLIT_ON):
        if key[1] in metadata:
            return metadata[key[1]]
    return "untitled"


def import_knowledge(docs_dir: Path, clear_existing: bool = False) -> None:
    try:
        vector_store = MilvusVectorStore(uri=agentConfig.milvus_uri)
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        sys.exit(1)

    memory_type = MemoryType.BUSINESS_KNOWLEDGE

    if clear_existing:
        try:
            vector_store.delete(memory_type=memory_type, where=None)
            logger.info("Cleared existing knowledge collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            sys.exit(1)

    doc_mapping = [
        ("product", "products.md"),
        ("faq", "faq.md"),
        ("term", "terms.md"),
        ("regulation", "regulations.md"),
        ("process", "process.md"),
    ]

    all_ids, all_texts, all_models = [], [], []
    version = "2025.04"

    for category, filename in doc_mapping:
        filepath = docs_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        chunks = chunk_document(filepath)
        if not chunks:
            logger.warning(f"No valid chunks in {filename}")
            continue

        print(f"===={category}:{chunks}")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{category}:{filename}:{i}"
            content = chunk["content"]
            all_ids.append(chunk_id)
            all_texts.append(content)

            model = BusinessKnowledge(
                id=chunk_id,
                text=content,
                source=filename,
                category=category,
                version=version,
                media_type="text/markdown",
                data_source_type="manual",
                file_url=""
            )
            all_models.append(model)

        logger.info(f"Processed {len(chunks)} chunks from {filename}")

    if not all_ids:
        logger.warning("No knowledge chunks to import")
        return

    logger.info(f"Importing {len(all_ids)} knowledge chunks ...")
    try:
        vector_store.add(
            memory_type=memory_type,
            ids=all_ids,
            texts=all_texts,
            models=all_models,
        )
        logger.info(f"Successfully imported {len(all_ids)} chunks")
    except Exception as e:
        logger.error(f"Failed to import knowledge: {e}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import business knowledge into Milvus")
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "knowledge"),
        help="Directory containing .md knowledge files (default: project-root/data/knowledge)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing knowledge collection before import",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import_knowledge(Path(args.docs_dir), args.clear)
    # client = MilvusVectorStore(agentConfig.milvus_uri)
    # results = client.get(MemoryType.BUSINESS_KNOWLEDGE)
    # print(results)
