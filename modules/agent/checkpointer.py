import sqlite3
import logging
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from config.models.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)

_sync_checkpointer: Optional[SqliteSaver] = None


def get_checkpointer(retrieval_cfg: RetrievalConfig) -> BaseCheckpointSaver:
    global _sync_checkpointer
    if _sync_checkpointer is None:
        db_path = retrieval_cfg.sqlite_db_path
        _ensure_db_directory(db_path)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout = 5000")

        logger.info("Build sync Checkpointer", extra={"db_path": db_path})

        _sync_checkpointer = SqliteSaver(conn)
        _sync_checkpointer.setup()
        logger.info("Sync Checkpointer initialized")
    return _sync_checkpointer


def _ensure_db_directory(db_path: str) -> None:
    db_file = Path(db_path)
    if db_file.suffix == ".db":
        parent = db_file.parent
    else:
        parent = db_file

    parent.mkdir(parents=True, exist_ok=True)
    logger.debug("The database directory is ready", extra={"path": str(parent)})