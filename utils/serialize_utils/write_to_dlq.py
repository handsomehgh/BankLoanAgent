# author hgh
# version 1.0
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def write_to_local_dlq(payload: dict, stream_name: str):
    dlq_path = Path(f"logs/dlq/{stream_name}.jsonl")
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dlq_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as write_error:
        logger.critical("本地 DLQ 写入失败: %s", write_error)