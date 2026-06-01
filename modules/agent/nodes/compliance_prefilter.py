# author hgh
# version 1.0
"""
public compliance pre-filter
function: performers compliance scanning before user input enters any business agent
policy: regex matching takes priority,LLM secondary review as a fallback,request_level deduplication cache
return: BLOCK/WARNING/PASS
"""
import logging
import re
import threading
from typing import Dict, Any, Optional, List

from cachetools import TTLCache
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import ComplianceAction, ComplianceSeverity, RegistryModules, ConfigFields
from config.global_constant.fields import CommonFields
from config.models.agent_config import SupervisorConfig
from config.models.memory_config import MemorySystemConfig
from config.prompts.compliance_fallback_prompt import COMPLIANCE_FALLBACK_PROMPT
from config.registry import ConfigRegistry
from modules.agent.constants import StateFields
from modules.agent.multi_agent_state import SupervisorState
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.module_services.chat_models import RobustLLM
from modules.tools.common_utils import assign_message_index
from utils.monitor_utils.metrics import compliance_block_total
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


class CompliancePrefilter:
    """public compliance pre-filter"""

    def __init__(
            self,
            memory_store: BaseMemoryStore,
            memory_config: MemorySystemConfig,
            registry: Optional[ConfigRegistry] = None,
            llm_client: Optional[RobustLLM] = None,
            seq_generator: SequenceGenerator = None
    ):
        self.memory_store = memory_store
        self.memory_config = memory_config
        self.supervisor_config = registry.get_config(RegistryModules.SUPERVISOR.value)
        self.llm_client = llm_client
        self.seq_generator = seq_generator

        self._cache = TTLCache(maxsize=200, ttl=5.0)
        self._cache_lock = threading.Lock()

    def _get_dedup_cache_key(self, user_id: str, query: str) -> str:
        import hashlib
        return hashlib.md5(f"{user_id}:{query}".encode()).hexdigest()

    def _dedup_lookup(self, user_id: str, query: str):
        key = self._get_dedup_cache_key(user_id, query)
        with self._cache_lock:
            return self._cache.get(key)

    def _dedup_store(self, user_id: str, query: str, result: dict):
        key = self._get_dedup_cache_key(user_id, query)
        with self._cache_lock:
            self._cache[key] = result

    def __call__(self, state: SupervisorState, config: RunnableConfig, ) -> Dict[str, Any]:
        # 1. get user input
        messages = state.get(StateFields.MESSAGES.value, [])
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content.strip()
                break
        if not user_query:
            logger.debug("User input is empty,let it pass directly")
            return {StateFields.COMPLIANCE_BLOCKED.value: False, StateFields.COMPLIANCE_WARNINGS.value: [],
                    StateFields.SHOULD_SKIP_SUPERVISOR.value: False}

        # 2. deduplication check
        user_id = state.get(StateFields.USER_ID.value, "unknown")
        session_id = config.get(ConfigFields.CONFIGURABLE.value, {}).get(ConfigFields.THREAD_ID.value)
        cached_result = self._dedup_lookup(user_id, user_query)
        if cached_result is not None:
            logger.debug("Use cached compliance scan result")
            return cached_result

        # 3. obtain compliance rules
        try:
            rules = self.memory_store.get_active_compliance_rules(20)
            logger.debug("Retrieved d% active compliance rules", len(rules))
        except Exception as e:
            logger.error("Failed to retrieval compliance rules: %s，downgrade release", e, exc_info=True)
            fallback_result = {
                StateFields.COMPLIANCE_BLOCKED.value: False,
                StateFields.COMPLIANCE_WARNINGS.value: ["合规规则加载异常，临时放行"],
                StateFields.SHOULD_SKIP_SUPERVISOR.value: False,
                StateFields.MANDATORY_APPENDS.value: [],
            }
            self._dedup_store(user_id, user_query, fallback_result)
            return fallback_result

        # 4. regex matching
        hit_rules = []
        for rule in rules:
            meta = rule.get(CommonFields.METADATA, rule)
            pattern = meta.get(CommonFields.PATTERN)
            if not pattern:
                continue
            try:
                if re.search(pattern, user_query, re.IGNORECASE):
                    hit_rules.append(rule)
            except Exception as e:
                logger.warning("Invalid regular expression rule_id=%s: %s", meta.get(CommonFields.RULE_ID), e)

        # 5. compliance check
        check_result = self._compliance_check(user_query, hit_rules, self.memory_config, self.supervisor_config,
                                              self.llm_client)
        blocked = check_result[StateFields.COMPLIANCE_BLOCKED.value]
        block_reason = check_result[StateFields.BLOCK_REASON.value]
        warnings = check_result[StateFields.COMPLIANCE_WARNINGS.value]
        mandatory_appends = check_result[StateFields.MANDATORY_APPENDS.value]

        # 6. assemble result
        if blocked:
            logger.warning("Compliance intercept,reason=%s, query='%s...'", block_reason, user_query[:80])
            compliance_block_total.labels(action='block', reason=block_reason or "未知原因").inc()
            compliance_response = AIMessage(
                content="不好意思，你的问题涉及违规，我暂时不能处理,\n\n"
                        "如果您有任何问题建议前往最近的银行网点咨询，谢谢合作😀"
            )
            assign_message_index(compliance_response, user_id, session_id, self.seq_generator)
            result = {
                StateFields.MESSAGES.value: [compliance_response],
                StateFields.COMPLIANCE_BLOCKED.value: blocked,
                StateFields.BLOCK_REASON.value: block_reason,
                StateFields.COMPLIANCE_WARNINGS.value: warnings,
                StateFields.MANDATORY_APPENDS.value: mandatory_appends,
                StateFields.SHOULD_SKIP_SUPERVISOR.value: True
            }
        else:
            logger.info("Compliance check passed，warnings=%d, appends=%d", len(warnings), len(mandatory_appends))
            result = {
                StateFields.COMPLIANCE_BLOCKED.value: False,
                StateFields.COMPLIANCE_WARNINGS.value: warnings,
                StateFields.MANDATORY_APPENDS.value: mandatory_appends,
                StateFields.SHOULD_SKIP_SUPERVISOR.value: False,
            }

        # 8. write to cache
        self._dedup_store(user_id, user_query, result)
        return result

    def _compliance_check(
            self,
            user_query: str,
            hit_rules: List[Dict[str, Any]],
            memory_config: MemorySystemConfig,
            supervisor_config: SupervisorConfig,
            llm_client: RobustLLM
    ) -> Dict[str, Any]:
        # 1. sorted by severity and priority
        severity_model = memory_config.compliance_severity
        hit_rules.sort(key=lambda r: (
            getattr(severity_model, r.get(CommonFields.SEVERITY, "medium"), 4),
            r.get(CommonFields.PRIORITY, 100)
        ))

        blocked = False
        block_reason = ""
        warnings = []
        mandatory_appends = []
        for rule in hit_rules:
            action = rule.get(CommonFields.ACTION, ComplianceAction.WARN)
            severity = rule.get(CommonFields.SEVERITY)
            if action == ComplianceAction.BLOCK and severity in (
                    ComplianceSeverity.CRITICAL, ComplianceSeverity.HIGH
            ):
                blocked = True
                block_reason = rule.get(CommonFields.RULE_NAME) or rule.get(CommonFields.RULE_ID, "合规规则")
                break
            elif action == ComplianceAction.WARN:
                desc = rule.get(CommonFields.DESCRIPTION)
                if desc:
                    warnings.append(desc)
            elif action == ComplianceAction.APPEND:
                template = rule.get(CommonFields.TEMPLATE)
                if template:
                    mandatory_appends.append(template)

        # 2. LLM secondary review fallback(only when no regex match and the configuration is enabled)
        enable_fallback = False
        if supervisor_config is not None:
            enable_fallback = supervisor_config.enable_compliance_llm_fallback

        if not hit_rules and enable_fallback and llm_client is not None:
            logger.debug("No match in regex, triggering LLM compliance secondary review")
            try:
                messages = COMPLIANCE_FALLBACK_PROMPT.invoke({"user_query": user_query[:800]}).to_messages()
                response = llm_client.invoke(messages)
                decision = response.content.strip().upper()
                logger.info("LLM compliance secondary result: %s", decision)
                if decision == "BLOCK":
                    blocked = True
                    block_reason = "LLM风险评估拦截"
                elif decision == "WARN":
                    warnings.append("您的请求可能涉及敏感内容，请注意合规使用")
            except Exception as e:
                logger.warning("Failed to LLM secondary review: %s release", e)

        return {
            StateFields.COMPLIANCE_BLOCKED.value: blocked,
            StateFields.BLOCK_REASON.value: block_reason,
            StateFields.COMPLIANCE_WARNINGS.value: warnings,
            StateFields.MANDATORY_APPENDS.value: mandatory_appends
        }
