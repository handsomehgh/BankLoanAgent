# author hgh
# version 1.0
from pydantic import BaseModel, Field


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    max_connections: int = 50
    socket_timeout: float = 0.5
    socket_connect_timeout: float = 1.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30

class GraphConfig(BaseModel):
    enabled: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_secure_password"
    database: str = "neo4j"
    query_timeout_ms: int =500
    max_connection_lifetime: int = 3600
    max_retry_attempts: int = 3

class MySQLConfig(BaseModel):
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "bank_agent"
    charset: str = "utf8mb4"
    pool_size: int = 10
    pool_recycle: int = 3600
    echo: bool = False

class PostgreSqlConfig(BaseModel):
    postgres_dsn: str = "postgresql://user:password@localhost:5432/bank_agent"
    postgres_pool_min_size: int = 2
    postgres_pool_max_size: int = 10
    postgres_pool_timeout: float = 5.0
    postgres_command_timeout: int = 30


class DataSourceConfig(BaseModel):
    redis: RedisConfig = Field(default_factory=RedisConfig)
    neo4j: GraphConfig = Field(default_factory=GraphConfig)
    mysql: MySQLConfig = Field(default_factory=MySQLConfig)
    postgresql: PostgreSqlConfig = Field(default_factory=PostgreSqlConfig)