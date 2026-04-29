# author hgh
# version 1.0
import sqlite3
import logging
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import config

logger = logging.getLogger(__name__)


def get_checkpointer() -> SqliteSaver:
    try:
        conn = sqlite3.connect(config.sqlite_db_path, check_same_thread=False)
        logger.info(f"SQLite checkpointer at {config.sqlite_db_path}")
        return SqliteSaver(conn)
    except Exception as e:
        logger.critical(f"Failed to initialize checkpointer: {e}")
        raise
