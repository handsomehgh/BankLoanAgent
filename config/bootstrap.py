import atexit
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, get_origin, Annotated, get_args

from prometheus_client import start_http_server
from sshtunnel import SSHTunnelForwarder

from config.container import ApplicationContainer
from config.settings import GlobalSettings
from config.models.bank_global_config import BankGlobalConfig
from config.models.agent_config import (
    SupervisorConfig, LoanAdvisorConfig, RiskAssessmentConfig,
    AfterLoanConfig, DirectReplyConfig, AgentExecutorConfig
)
from config.models.memory_config import MemorySystemConfig
from config.models.retrieval_config import RetrievalConfig
from config.models.llm_config import LLMConfig
from config.models.cache_config import CacheConfig
from config.models.datasource_config import DataSourceConfig
from config.models.tool_config import ToolRegistryConfig
from config.global_constant.constants import RegistryModules, CacheNamespace
from config.context_settings import set_enum_strictness
from infra.cache.cache_registry import cache_register
from infra.database.mysql_manager import DatabaseManager
from infra.database.redis_manager import RedisManager
from utils.cache_utils.cache_decorator import set_cache_container
from utils.logging_config import setup_logging
from modules.agent.multi_graph import MultiAgentGraphBuilder
from modules.agent.human_handoff.handoff_timeout_monitor import HandoffTimeoutMonitor
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from langgraph.graph import StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    """应用启动后暴露给外部的核心运行时对象"""
    graph: StateGraph
    memory_store: BaseMemoryStore
    redis_manager: RedisManager


class Bootstrapper:
    """应用启动器，负责按阶段初始化所有全局组件并返回运行时对象"""

    def __init__(self):
        self._container: Optional[ApplicationContainer] = None
        self._registry = None
        self._graph = None
        self._runtime: Optional[AppRuntime] = None
        self._started = False
        self._ssh_tunnel: Optional[SSHTunnelForwarder] = None

    # ==================================================================
    # 公共入口
    # ==================================================================
    def start(self) -> AppRuntime:
        """启动应用，返回 AppRuntime"""
        if self._started:
            return self._runtime

        logger.info("=" * 50)
        logger.info("Application starting...")
        logger.info("=" * 50)

        # 创建容器和配置注册中心
        self._container = ApplicationContainer()
        self._registry = self._container.config_registry()

        self._phase_load_config()                 # 阶段1：配置
        self._phase_init_infrastructure()         # 阶段2：基础设施（含SSH隧道）
        self._phase_register_tools_and_skills()   # 阶段3：工具与技能
        self._phase_build_graph()                 # 阶段4：构建图
        self._phase_start_background_services()   # 阶段5：后台服务

        self._started = True
        logger.info("=" * 50)
        logger.info("Application started successfully")
        logger.info("=" * 50)
        return self._runtime

    @property
    def runtime(self) -> AppRuntime:
        if not self._started:
            raise RuntimeError("Bootstrapper not started")
        return self._runtime

    # ==================================================================
    # 阶段1：加载配置
    # ==================================================================
    def _phase_load_config(self):
        logger.info("[Phase 1/5] Loading configuration...")

        # 环境变量
        settings = GlobalSettings()

        # 注册所有配置模型并加载
        self._register_configs(self._registry, PROJECT_ROOT)
        self._registry.load_all()

        # 注入敏感信息
        self._inject_sensitive(self._registry, settings)

        # 严格模式
        memory_config = self._registry.get_config(RegistryModules.MEMORY_SYSTEM)
        set_enum_strictness(memory_config.strict_enum_validation)

        # 启动热更新
        self._registry.start_hot_reload()

        logger.info("[Phase 1/5] Configuration loaded successfully")

    # ==================================================================
    # 阶段2：初始化基础设施
    # ==================================================================
    def _phase_init_infrastructure(self):
        logger.info("[Phase 2/5] Initializing infrastructure...")

        # 1. 建立SSH隧道（如果需要连接远程服务）
        self._setup_ssh_tunnels()

        # 2. 触发容器中所有单例的初始化（除工具注册，它在阶段3）
        self._container.init_resources()

        # 3. 缓存管理器注册
        self._register_caches()

        # 4. 日志配置
        llm_config = self._registry.get_config(RegistryModules.LLM)
        setup_logging(log_level=llm_config.log_level)

        # 6. Prometheus
        # self._start_prometheus()

        logger.info("[Phase 2/5] Infrastructure initialized successfully")

    def _setup_ssh_tunnels(self):
        """使用SSH隧道将远程服务器端口映射到本地（8000→8080, 8001→8081, ...）"""
        # 读取环境变量
        ssh_host = os.getenv("SSH_HOST")
        ssh_port = int(os.getenv("SSH_PORT", "22"))
        ssh_user = os.getenv("SSH_USER")
        ssh_key = os.getenv("SSH_KEY_PATH")
        ssh_key_password = os.getenv("SSH_KEY_PASSWORD", "")

        # 检查必要配置是否存在
        if not all([ssh_host, ssh_user, ssh_key, ssh_key_password]):
            logger.info("SSH tunnel configuration incomplete, skipping.")
            return

        # 定义端口映射：本地 8000-8005 → 远程 localhost:8080-8085
        local_start = 8000
        remote_start = 8080
        num_tunnels = 1

        try:
            remote_bind_addresses = [
                ("localhost", remote_start + i) for i in range(num_tunnels)
            ]
            local_bind_addresses = [
                ("127.0.0.1", local_start + i) for i in range(num_tunnels)
            ]

            # 创建 SSH 隧道对象
            self._ssh_tunnel = SSHTunnelForwarder(
                (ssh_host, ssh_port),
                ssh_username=ssh_user,
                ssh_pkey=ssh_key,
                ssh_private_key_password=ssh_key_password,
                remote_bind_addresses=remote_bind_addresses,
                local_bind_addresses=local_bind_addresses,
                set_keepalive=15.0,
            )

            # 启动隧道
            self._ssh_tunnel.start()
            logger.info(
                "SSH tunnels established: local %s -> remote %s:%s",
                [local[1] for local in local_bind_addresses],
                ssh_host,
                [remote[1] for remote in remote_bind_addresses]
            )

            # 注册程序退出时自动关闭隧道
            atexit.register(self._stop_ssh_tunnels)

        except Exception as e:
            logger.error("Failed to start SSH tunnels: %s", e)
            self._ssh_tunnel = None

    def _stop_ssh_tunnels(self):
        """停止 SSH 隧道（由 atexit 注册调用）"""
        if hasattr(self, '_ssh_tunnel') and self._ssh_tunnel:
            try:
                self._ssh_tunnel.stop()
                logger.info("SSH tunnels stopped.")
            except Exception as e:
                logger.warning("Error while stopping SSH tunnels: %s", e)

    # ==================================================================
    # 阶段3：注册工具与技能
    # ==================================================================
    def _phase_register_tools_and_skills(self):
        logger.info("[Phase 3/5] Registering tools and skills...")

        # 工具注册：扫描并验证
        tool_registry = self._container.tool_registry()
        tool_registry.scan_and_register()
        tool_registry.validate_against_config()

        # 技能注册（从 container.py 移出，避免逻辑代码污染容器定义）
        self._register_skills(tool_registry)

        logger.info("[Phase 3/5] Tools and skills registered successfully")

    # ==================================================================
    # 阶段4：构建 Agent 图
    # ==================================================================
    def _phase_build_graph(self):
        logger.info("[Phase 4/5] Building Agent graph...")
        self._bind_tool_injections()

        builder = MultiAgentGraphBuilder(self._container)
        self._graph = builder.build()

        logger.info("[Phase 4/5] Agent graph built successfully")

    # ==================================================================
    # 阶段5：启动后台服务
    # ==================================================================
    def _phase_start_background_services(self):
        logger.info("[Phase 5/5] Starting background services...")

        # 消费者线程
        self._start_consumers()

        # 超时监控
        timeout_monitor = HandoffTimeoutMonitor(
            self._graph, self._container.redis_manager()
        )
        timeout_monitor.start()

        # 缓存容器设置
        set_cache_container(self._container)

        # 封装运行时对象
        self._runtime = AppRuntime(
            graph=self._graph,
            memory_store=self._container.memory_store(),
            redis_manager=self._container.redis_manager()
        )

        logger.info("[Phase 5/5] Background services started successfully")

    # ==================================================================
    # 辅助方法：配置注册
    # ==================================================================
    def _register_configs(self, registry, root):
        registry.register_model(RegistryModules.MEMORY_SYSTEM.value, MemorySystemConfig,
                                root / "config/rules/memory_system_config.yaml")
        registry.register_model(RegistryModules.RETRIEVAL.value, RetrievalConfig,
                                root / "config/rules/retrieval_config.yaml")
        registry.register_model(RegistryModules.LLM.value, LLMConfig,
                                root / "config/rules/llm_config.yaml")
        registry.register_model(RegistryModules.CACHE.value, CacheConfig,
                                root / "config/rules/cache.yaml")
        registry.register_model(RegistryModules.DATASOURCE.value, DataSourceConfig,
                                root / "config/rules/datasource_config.yaml")
        registry.register_model(RegistryModules.SUPERVISOR.value, SupervisorConfig,
                                root / "config/rules/supervisor.yaml")
        registry.register_model(RegistryModules.LOAN_ADVISOR.value, LoanAdvisorConfig,
                                root / "config/rules/loan_advisor.yaml")
        registry.register_model(RegistryModules.RISK_ASSESSMENT.value, RiskAssessmentConfig,
                                root / "config/rules/risk_assessment.yaml")
        registry.register_model(RegistryModules.AFTER_LOAN.value, AfterLoanConfig,
                                root / "config/rules/after_loan.yaml")
        registry.register_model(RegistryModules.TOOL_REGISTRY.value, ToolRegistryConfig,
                                root / "config/rules/tool_registry.yaml")
        registry.register_model(RegistryModules.BANK_GLOBAL_CONFIG.value, BankGlobalConfig,
                                root / "config/rules/bank_global_config.yaml")
        registry.register_model(RegistryModules.DIRECT_REPLY.value, DirectReplyConfig,
                                root / "config/rules/direct_reply.yaml")
        registry.register_model(RegistryModules.AGENT_EXECUTOR.value, AgentExecutorConfig,
                                root / "config/rules/agent_executor.yaml")

    def _inject_sensitive(self, registry, settings):
        llm_cfg = registry.get_config(RegistryModules.LLM.value)
        llm_cfg.deepseek_api_key = settings.deepseek_api_key
        llm_cfg.alibaba_api_key = settings.alibaba_api_key
        llm_cfg.log_level = settings.log_level
        registry.update_config(RegistryModules.LLM.value, llm_cfg)

    # ==================================================================
    # 辅助方法：缓存注册
    # ==================================================================
    def _register_caches(self):
        cache_register(CacheNamespace.RAG.value, self._container.rag_cache())
        cache_register(CacheNamespace.COMPLIANCE.value, self._container.compliance_cache())
        cache_register(CacheNamespace.PROFILE_SUMMARY.value, self._container.profile_summary_cache())
        cache_register(CacheNamespace.RECENT_INTERACTION.value, self._container.interaction_cache())
        cache_register(CacheNamespace.LPR.value, self._container.lpr_cache())

    # ==================================================================
    # 辅助方法：工具依赖绑定
    # ==================================================================
    def _bind_tool_injections(self):
        import inspect
        from typing import get_type_hints
        from langchain_core.tools import InjectedToolArg
        from modules.module_services.lpr_data_service import LPRDataService
        from config.models.bank_global_config import BankGlobalConfig
        from modules.retrieval.retrieval_service import RetrievalService

        registry = self._container.tool_registry()
        type_mapping = {
            LPRDataService: self._container.lpr_service,
            BankGlobalConfig: self._container.bank_global_config,
            RetrievalService: self._container.knowledge_retriever,
            DatabaseManager: self._container.db_manager
        }
        for versions in registry._tools.values():
            for tool in versions.values():
                func = getattr(tool, 'func', None)
                if not func:
                    continue
                sig = inspect.signature(func)
                hints = get_type_hints(func, include_extras=True)
                injected = {}
                injected_param_names = []
                for name, param in sig.parameters.items():
                    ann = hints.get(name)
                    if ann:
                        # 如果是有 Annotated 的类型
                        if get_origin(ann) is Annotated:
                            args = get_args(ann)
                            if len(args) >= 2 and args[1] == InjectedToolArg:
                                dep_type = args[0]
                                dep = type_mapping.get(dep_type)
                                if dep:
                                    injected[name] = dep() if callable(dep) else dep
                                else:
                                    injected_param_names.append(name)
                tool._injected_kwargs = injected
                tool._injected_params = injected_param_names

    # ==================================================================
    # 辅助方法：技能注册（从 container.py 移出）
    # ==================================================================
    def _register_skills(self, tool_registry):
        from config.skills_loader import load_skill_configs
        from modules.skills.skill_factory import create_tool_from_skill

        skill_configs = load_skill_configs(PROJECT_ROOT / "config/skills")
        skill_registry = self._container.skill_registry()

        for cfg in skill_configs:
            skill_registry.register(cfg)
            tool = create_tool_from_skill(cfg)
            tool_registry.register(tool)

        logger.info("Loaded %d skill configs and registered as tools", len(skill_configs))

    # ==================================================================
    # 辅助方法：Prometheus
    # ==================================================================
    def _start_prometheus(self):
        port = 9090
        try:
            start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning("Port %d already in use, trying to free it.", port)
                import subprocess, signal
                try:
                    out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
                    for pid in out.strip().split():
                        os.kill(int(pid), signal.SIGTERM)
                except Exception:
                    pass
                start_http_server(port)
                logger.info("Prometheus metrics server started on port %d after cleanup.", port)
            else:
                raise

    # ==================================================================
    # 辅助方法：消费者线程
    # ==================================================================
    def _start_consumers(self):
        interaction_consumer = self._container.interaction_log_consumer()
        sub_interaction_consumer = self._container.sub_interaction_log_consumer()
        user_profile_consumer = self._container.user_profile_consumer()
        threading.Thread(target=interaction_consumer.process, daemon=True, name="interaction-consumer").start()
        threading.Thread(target=sub_interaction_consumer.process, daemon=True, name="sub-interaction-consumer").start()
        threading.Thread(target=user_profile_consumer.process, daemon=True, name="user-profile-consumer").start()


# ==================================================================
# 全局单例
# ==================================================================
_BOOTSTRAPPER = Bootstrapper()


def get_bootstrapper() -> Bootstrapper:
    return _BOOTSTRAPPER