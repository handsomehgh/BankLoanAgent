# author hgh
# version 1.0
"""
Fan-out distribution node: parallel invocation of multiple sub-Agents, with support for timeout fallback
"""
import logging
from typing import Dict, Any

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_core.runnables import RunnableConfig

from modules.agent.constants import StateFields, AgentName
from modules.agent.multi_agent_state import SupervisorState, AgentResponse

logger = logging.getLogger(__name__)

class FanoutDispatcher:
    def __init__(self,agent_time_out: int = 60,agent_graphs: Dict[str,Any] = None):
        """
        agent_graphs: subgraph name -> complied subgraph instance
        """
        self.agent_graphs = agent_graphs
        self.agent_time_out = agent_time_out


    def __call__(self,state: SupervisorState,config: RunnableConfig):
        next_agents = state.get(StateFields.NEXT_AGENTS.value, [])
        agent_contexts = state.get(StateFields.AGENT_CONTEXT.value, {})
        if not next_agents:
            logger.warning("[FanoutDispatcher] no next agents found")
            return {}

        agent_responses = {}
        sub_messages = {}
        trigger_handoff = False
        with ThreadPoolExecutor(max_workers=len(next_agents)) as executor:
            future_to_agent = {}
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
                future = executor.submit(subgraph.invoke,sub_state,config)
                future_to_agent[future] = agent_name

            for future in future_to_agent:
                agent_name = future_to_agent[future]
                try:
                    sub_result = future.result(timeout=self.agent_time_out)
                    final_resp = sub_result[StateFields.FINAL_RESPONSE.value]
                    if final_resp:
                        agent_responses[agent_name] = final_resp
                    msgs = sub_result.get(StateFields.INTERNAL_MESSAGES.value, [])
                    sub_messages.setdefault(agent_name, []).extend(msgs)

                    if agent_name == AgentName.RISK_ASSESSMENT.value:
                        trigger_handoff = trigger_handoff or sub_result.get(
                            StateFields.TRIGGER_HUMAN_HANDOFF.value, False
                        )
                except FuturesTimeoutError:
                    logger.error("[FanoutDispatcher] Agent %s timed out", agent_name)
                    fallback = AgentResponse(content="抱歉，该服务暂时无响应，请稍后再试。")
                    agent_responses[agent_name] = fallback
                    sub_messages[agent_name] = []
                except Exception as e:
                    logger.exception("[FanoutDispatcher] Agent %s execution failed: %s", agent_name, e)
                    fallback = AgentResponse(content="抱歉，处理您的请求时出现错误。")
                    agent_responses[agent_name] = fallback
                    sub_messages[agent_name] = []

        return {
            StateFields.AGENT_RESPONSES.value: agent_responses,
            StateFields.SUB_MESSAGES.value: sub_messages,
            StateFields.TRIGGER_HUMAN_HANDOFF.value: trigger_handoff,
            StateFields.NEXT_AGENTS.value: [],
            StateFields.AGENT_CONTEXT.value: {},
        }


        
