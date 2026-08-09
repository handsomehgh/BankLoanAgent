# author hgh
# version 1.1
"""
Fan-out distribution node: parallel invocation of multiple sub-Agents,
with support for timeout fallback (async version)
"""
import asyncio
import logging
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from modules.agent.constants import StateFields, AgentName
from modules.agent.multi_agent_state import SupervisorState, AgentResponse

logger = logging.getLogger(__name__)


class FanoutDispatcher:
    def __init__(self, agent_time_out: int = 60, agent_graphs: Dict[str, Any] = None):
        """
        agent_graphs: subgraph name -> compiled subgraph instance
        """
        self.agent_graphs = agent_graphs
        self.agent_time_out = agent_time_out

    async def _invoke_with_timeout(self, subgraph, sub_state, config, agent_name):
        """单个 subgraph 的超时包装"""
        try:
            return await asyncio.wait_for(
                subgraph.ainvoke(sub_state, config=config),
                timeout=self.agent_time_out
            )
        except asyncio.TimeoutError:
            logger.error("[FanoutDispatcher] Agent %s timed out", agent_name)
            return {
                StateFields.FINAL_RESPONSE.value: AgentResponse(content="抱歉，该服务暂时无响应，请稍后再试。"),
                StateFields.INTERNAL_MESSAGES.value: [],
            }

    async def __call__(self, state: SupervisorState, config: RunnableConfig):
        next_agents = state.get(StateFields.NEXT_AGENTS.value, [])
        agent_contexts = state.get(StateFields.AGENT_CONTEXT.value, {})
        if not next_agents:
            logger.warning("[FanoutDispatcher] no next agents found")
            return {}

        agent_responses = {}
        sub_messages = {}
        trigger_handoff = False

        tasks = []
        task_to_agent = {}
        for agent_name in next_agents:
            if agent_name not in self.agent_graphs:
                logger.warning("[FanoutDispatcher] agent %s not registered, skipping", agent_name)
                continue
            subgraph = self.agent_graphs[agent_name]
            ctx = agent_contexts.get(agent_name)
            if not ctx:
                logger.warning("[FanoutDispatcher] fanout: lack %s AgentContext, using fallback", agent_name)
                agent_responses[agent_name] = AgentResponse(content="抱歉，该服务暂时不可用，请稍后再试。")
                sub_messages[agent_name] = []
                continue
            sub_state = {StateFields.AGENT_CONTEXT.value: ctx}
            task = asyncio.create_task(self._invoke_with_timeout(subgraph, sub_state, config, agent_name))
            tasks.append(task)
            task_to_agent[task] = agent_name

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(tasks, results):
            agent_name = task_to_agent[task]
            if isinstance(result, Exception):
                logger.exception("[FanoutDispatcher] Agent %s execution failed: %s", agent_name, result)
                fallback = AgentResponse(content="抱歉，处理您的请求时出现错误。")
                agent_responses[agent_name] = fallback
                sub_messages[agent_name] = []
                continue

            final_resp = result.get(StateFields.FINAL_RESPONSE.value)
            if final_resp:
                agent_responses[agent_name] = final_resp
            msgs = result.get(StateFields.INTERNAL_MESSAGES.value, [])
            sub_messages.setdefault(agent_name, []).extend(msgs)

            if agent_name == AgentName.RISK_ASSESSMENT.value:
                trigger_handoff = trigger_handoff or result.get(
                    StateFields.TRIGGER_HUMAN_HANDOFF.value, False
                )

        return {
            StateFields.AGENT_RESPONSES.value: agent_responses,
            StateFields.SUB_MESSAGES.value: sub_messages,
            StateFields.TRIGGER_HUMAN_HANDOFF.value: trigger_handoff,
            StateFields.NEXT_AGENTS.value: [],
            StateFields.AGENT_CONTEXT.value: {},
        }
