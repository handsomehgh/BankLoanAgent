# author hgh
# version 1.0
from pydantic import BaseModel


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