# author hgh
# version 1.0
# infra/database/database_manager.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config.models.datasource_config import MySQLConfig

class DatabaseManager:
    """数据库管理器：负责引擎和会话工厂的创建与销毁"""

    def __init__(self, mysql_config: MySQLConfig):
        self.config = mysql_config
        self._engine = None
        self._session_factory = None
        self._init_engine()

    def _init_engine(self):
        url = (
            f"mysql+pymysql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
            f"?charset={self.config.charset}"
        )
        self._engine = create_engine(
            url,
            pool_size=self.config.pool_size,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo,
        )
        self._session_factory = sessionmaker(bind=self._engine)

    @property
    def engine(self):
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        return self._session_factory

    def create_session(self) -> Session:
        """创建一个新的数据库会话，调用方负责关闭"""
        return self._session_factory()

    def dispose(self):
        """释放连接池"""
        if self._engine:
            self._engine.dispose()