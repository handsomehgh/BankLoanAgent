# config/bootstrap.py
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from prometheus_client import start_http_server

from config.container import ApplicationContainer
from config.settings import GlobalSettings
from config.models.bank_global_config import BankGlobalConfig
from config.models.agent_config import SupervisorConfig, LoanAdvisorConfig, RiskAssessmentConfig, AfterLoanConfig, \
    DirectReplyConfig, AgentExecutorConfig
from config.models.memory_config import MemorySystemConfig
from config.models.retrieval_config import RetrievalConfig
from config.models.llm_config import LLMConfig
from config.models.cache_config import CacheConfig
from config.models.datasource_config import DataSourceConfig
from config.models.tool_config import ToolRegistryConfig
from config.global_constant.constants import RegistryModules, CacheNamespace
from config.context_settings import set_enum_strictness
from infra.cache.cache_registry import cache_register
from utils.cache_utils.cache_decorator import set_cache_container
from utils.logging_config import setup_logging
from modules.agent.multi_graph import MultiAgentGraphBuilder
from modules.agent.human_handoff.handoff_timeout_monitor import HandoffTimeoutMonitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


class Bootstrapper:
    """应用启动器，负责初始化所有全局组件并缓存单例"""

    def __init__(self):
        self._container: Optional[ApplicationContainer] = None
        self._graph = None
        self._memory_store = None
        self._started = False

    def start(self):
        if self._started:
            return
        settings = GlobalSettings()
        container = ApplicationContainer()
        registry = container.config_registry()

        # 1. 注册并加载配置
        self._register_configs(registry, PROJECT_ROOT)
        registry.load_all()

        # 2. 严格模式
        memory_config = registry.get_config(RegistryModules.MEMORY_SYSTEM)
        set_enum_strictness(memory_config.strict_enum_validation)

        # 3. 注入敏感信息
        self._inject_sensitive(registry, settings)

        # 4. 热更新监控
        registry.start_hot_reload()

        # 5. 缓存管理器注册
        cache_register(CacheNamespace.RAG.value, container.rag_cache())
        cache_register(CacheNamespace.COMPLIANCE.value, container.compliance_cache())
        cache_register(CacheNamespace.PROFILE_SUMMARY.value, container.profile_summary_cache())
        cache_register(CacheNamespace.RECENT_INTERACTION.value, container.interaction_cache())
        cache_register(CacheNamespace.LPR.value, container.lpr_cache())

        # 6. 工具依赖绑定
        self._bind_tool_injections(container)
        container.init_resources()

        # 7. 日志配置
        llm_config = registry.get_config(RegistryModules.LLM)
        setup_logging(log_level=llm_config.log_level)

        # 8. Prometheus 服务器
        self._start_prometheus()

        # 9. 构建 Agent Graph
        builder = MultiAgentGraphBuilder(container)
        graph = builder.build()

        # 10. 启动消费者线程
        self._start_consumers(container)

        # 11. 超时监控
        timeout_monitor = HandoffTimeoutMonitor(graph, container.redis_manager())
        timeout_monitor.start()

        # 12. 设置容器
        set_cache_container(container)

        self._container = container
        self._graph = graph
        self._memory_store = container.memory_store()
        self._started = True

    @property
    def container(self):
        if not self._started:
            raise RuntimeError("Bootstrapper not started")
        return self._container

    @property
    def graph(self):
        if not self._started:
            raise RuntimeError("Bootstrapper not started")
        return self._graph

    @property
    def memory_store(self):
        if not self._started:
            raise RuntimeError("Bootstrapper not started")
        return self._memory_store

    # 以下私有方法与原 app.py 中的函数对应
    def _register_configs(self, registry, root):
        registry.register_model(RegistryModules.MEMORY_SYSTEM.value, MemorySystemConfig,
                                root / "config/rules/memory_system_config.yaml")
        registry.register_model(RegistryModules.RETRIEVAL, RetrievalConfig, root / "config/rules/retrieval_config.yaml")
        registry.register_model(RegistryModules.LLM, LLMConfig, root / "config/rules/llm_config.yaml")
        registry.register_model(RegistryModules.CACHE, CacheConfig, root / "config/rules/cache.yaml")
        registry.register_model(RegistryModules.DATASOURCE, DataSourceConfig,
                                root / "config/rules/datasource_config.yaml")
        registry.register_model(RegistryModules.SUPERVISOR, SupervisorConfig, root / "config/rules/supervisor.yaml")
        registry.register_model(RegistryModules.LOAN_ADVISOR, LoanAdvisorConfig,
                                root / "config/rules/loan_advisor.yaml")
        registry.register_model(RegistryModules.RISK_ASSESSMENT, RiskAssessmentConfig,
                                root / "config/rules/risk_assessment.yaml")
        registry.register_model(RegistryModules.AFTER_LOAN, AfterLoanConfig, root / "config/rules/after_loan.yaml")
        registry.register_model(RegistryModules.TOOL_REGISTRY, ToolRegistryConfig,
                                root / "config/rules/tool_registry.yaml")
        registry.register_model(RegistryModules.BANK_GLOBAL_CONFIG, BankGlobalConfig,
                                root / "config/rules/bank_global_config.yaml")
        registry.register_model(RegistryModules.DIRECT_REPLY, DirectReplyConfig,
                                root / "config/rules/direct_reply.yaml")
        registry.register_model(RegistryModules.AGENT_EXECUTOR, AgentExecutorConfig,
                                root / "config/rules/agent_executor.yaml")

    def _inject_sensitive(self, registry, settings):
        llm_cfg = registry.get_config(RegistryModules.LLM)
        llm_cfg.deepseek_api_key = settings.deepseek_api_key
        llm_cfg.alibaba_api_key = settings.alibaba_api_key
        llm_cfg.log_level = settings.log_level
        registry.update_config(RegistryModules.LLM, llm_cfg)

    def _bind_tool_injections(self, container):
        import inspect
        from typing import get_type_hints
        from langchain_core.tools import InjectedToolArg
        from modules.module_services.lpr_data_service import LPRDataService
        from config.models.bank_global_config import BankGlobalConfig
        from modules.retrieval.retrieval_service import RetrievalService

        registry = container.tool_registry()
        type_mapping = {
            LPRDataService: container.lpr_service,
            BankGlobalConfig: container.bank_global_config,
            RetrievalService: container.knowledge_retriever
        }
        for versions in registry._tools.values():
            for tool in versions.values():
                func = getattr(tool, 'func', None)
                if not func: continue
                sig = inspect.signature(func)
                hints = get_type_hints(func, include_extras=True)
                injected = {}
                injected_param_names = []
                for name, param in sig.parameters.items():
                    ann = hints.get(name)
                    if ann and hasattr(ann, '__metadata__') and InjectedToolArg in ann.__metadata__:
                        dep_type = ann.__origin__
                        dep = type_mapping.get(dep_type)
                        if dep:
                            injected[name] = dep() if callable(dep) else dep
                        else:
                            injected_param_names.append(name)
                tool._injected_kwargs = injected
                tool._injected_params = injected_param_names

    def _start_prometheus(self):
        port = 9090
        try:
            start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning("Port %d already in use, trying to free it.", port)
                try:
                    import subprocess, signal
                    out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
                    for pid in out.strip().split():
                        os.kill(int(pid), signal.SIGTERM)
                except Exception:
                    pass
                start_http_server(port)
                logger.info("Prometheus metrics server started on port %d after cleanup.", port)
            else:
                raise

    def _start_consumers(self, container):
        interaction_consumer = container.interaction_log_consumer()
        sub_interaction_consumer = container.sub_interaction_log_consumer()
        threading.Thread(target=interaction_consumer.process, daemon=True, name="interaction-consumer").start()
        threading.Thread(target=sub_interaction_consumer.process, daemon=True, name="sub-interaction-consumer").start()


# 全局单例
_BOOTSTRAPPER = Bootstrapper()


def get_bootstrapper() -> Bootstrapper:
    return _BOOTSTRAPPER
