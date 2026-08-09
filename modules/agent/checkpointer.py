import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config.models.datasource_config import DataSourceConfig

logger = logging.getLogger(__name__)

class CheckpointerInitError(Exception):
    pass

@asynccontextmanager
async def create_async_postgres_checkpointer(
    config: DataSourceConfig,
) -> AsyncGenerator[AsyncPostgresSaver, None]:
    dsn = config.postgresql.postgres_dsn
    if not dsn:
        raise CheckpointerInitError("PostgreSQL DSN is not configured")

    try:
        logger.info("Connecting to PostgreSQL with DSN: %s", dsn)
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()
            logger.info("PostgreSQL checkpoint tables verified/created")
            yield saver
    except Exception as e:
        logger.error("Failed to initialize PostgreSQL checkpointer: %s", e, exc_info=True)
        raise CheckpointerInitError(f"Checkpointer initialization failed: {e}") from e