from xml.etree.ElementTree import register_namespace

import yaml
import threading
from pathlib import Path
from typing import Type, Optional
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

logger = logging.getLogger(__name__)

class ConfigRegistry:
    _instance: Optional['ConfigRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._configs = {}
                    cls._instance._models = {}
                    cls._instance._file_paths = {}
        return cls._instance

    def register_model(self, module: str, model_cls: Type[BaseModel], yaml_path: Path):
        self._models[module] = model_cls
        self._file_paths[module] = yaml_path

    def load_all(self):
        print(f"models------------{self._models}")
        for module in self._models:
            self._load_module(module)

    def _load_module(self, module: str):
        path = self._file_paths[module]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            config = self._models[module](**data)
            with self._lock:
                self._configs[module] = config
            logger.info(f"配置模块 [{module}] 已加载/更新，版本：{getattr(config, 'version', 'N/A')}")
        except Exception as e:
            logger.error(f"加载配置模块 [{module}] 失败，路径 {path}: {e}")

    def get_config(self, module: str) -> BaseModel:
        with self._lock:
            if module not in self._configs:
                raise KeyError(f"未注册的配置模块: {module}")
            return self._configs[module]

    def update_config(self, module: str, new_config: BaseModel):
        with self._lock:
            self._configs[module] = new_config
        logger.info(f"配置模块 [{module}] 已手动热更新")

    def start_hot_reload(self) -> Observer:
        event_handler = ConfigFileEventHandler(self)
        observer = Observer()
        directories = set()
        for path in self._file_paths.values():
            directories.add(path.parent)
        for d in directories:
            observer.schedule(event_handler, path=str(d), recursive=False)
        observer.start()
        logger.info("配置文件热更新监控已启动")
        return observer

class ConfigFileEventHandler(FileSystemEventHandler):
    def __init__(self, registry: ConfigRegistry):
        self.registry = registry

    def on_modified(self, event):
        if event.is_directory:
            return
        filepath = Path(event.src_path)
        for module, path in self.registry._file_paths.items():
            if filepath == path:
                logger.info(f"检测到配置文件变更: {filepath}，开始热更新模块 [{module}]")
                self.registry._load_module(module)
                print(self.registry.get_config(module))
                break