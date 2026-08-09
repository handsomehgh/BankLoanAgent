"""
BankLoanAgent — FastAPI Application Entry Point
SSE streaming + REST API (replaces Streamlit app.py)
"""
import logging
import uuid
from typing import AsyncGenerator

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from api.api_models import (
    ChatRequest, ChatResponse,
    HandoffResolveRequest,
    SessionResponse, HistoryResponse, MemoryResponse,
)
from api.sse_formatter import format_sse
from config.bootstrap import AppRuntime
from config.global_constant.constants import ConfigFields, MemoryType
from modules.memory.memory_constant.constants import MemoryStatus
from utils.logging_config import set_log_context

logger = logging.getLogger(__name__)

# Filter keywords for astream_events: only capture tokens from these node names
RESPONSE_NODE_KEYWORDS = ["response", "fanout", "direct_reply"]


# ==================== SSE Event Generator ====================

async def _stream_events(
        graph,
        input_state: dict,
        config: dict,
        thread_id: str,
) -> AsyncGenerator[str, None]:
    """
    Core SSE event generator with heartbeat.
    Drives graph execution via astream_events(v2) and yields SSE-formatted frames.
    Sends heartbeat comments every 15s to prevent proxy/connection timeouts.
    """
    queue: asyncio.Queue = asyncio.Queue()
    heartbeat_interval = 15

    async def _heartbeat():
        """Periodically push heartbeat frames into the queue."""
        while True:
            await asyncio.sleep(heartbeat_interval)
            await queue.put(": heartbeat\n\n")

    async def _run_graph():
        """Drive graph execution and push SSE events into the queue."""
        try:
            async for event in graph.astream_events(input_state, config, version="v2"):
                kind = event.get("event", "")

                # ── LLM Token Stream ──
                if kind == "on_chat_model_stream":
                    print(f"event--------------------{event}")

                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                        continue
                    token = chunk.content

                    if token:
                        node = event.get("metadata", {}).get("langgraph_node", "")
                        if any(kw in node for kw in RESPONSE_NODE_KEYWORDS):
                            await queue.put(format_sse("token", {"content": token}))

            # Stream completed normally
            await queue.put(format_sse("done", {"status": "completed"}))

        except GraphInterrupt as e:
            logger.info("[SSE] GraphInterrupt caught: %s", e)
            await queue.put(format_sse("handoff", {
                "message": str(e) if str(e) else "需要转接人工客服",
                "thread_id": thread_id,
            }))

        except Exception as e:
            logger.exception("[SSE] Unexpected error during streaming")
            await queue.put(format_sse("error", {"message": str(e)}))

        finally:
            await queue.put(None)

    heartbeat_task = asyncio.create_task(_heartbeat())
    graph_task = asyncio.create_task(_run_graph())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        # Ensure graph task is also cleaned up
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except asyncio.CancelledError:
                pass


# ==================== Application Lifecycle ====================

from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from config.global_constant.constants import RegistryModules
from modules.agent.checkpointer import create_async_postgres_checkpointer
from config.bootstrap import get_bootstrapper

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[FastAPI] Application starting up...")
    bootstrapper = get_bootstrapper()
    loop = asyncio.get_running_loop()

    # 1. 同步初始化阶段（配置、基础设施、工具注册）
    await asyncio.to_thread(bootstrapper.start_sync, loop)

    # 2. 获取数据库配置（需要暴露 registry）
    registry = bootstrapper._registry
    db_cfg = registry.get_config(RegistryModules.DATASOURCE)

    # 3. 创建 checkpointer（连接池自动管理）
    async with create_async_postgres_checkpointer(db_cfg) as checkpointer:
        runtime = await bootstrapper.finish_build_async(checkpointer)
        app.state.runtime = runtime

        logger.info("[FastAPI] Application ready — graph and all services initialized")
        yield

    # 5. 退出 with 块时，连接池自动关闭
    logger.info("[FastAPI] Application shutting down...")
# ==================== FastAPI Application ====================

app = FastAPI(
    title="BankLoanAgent API",
    description="银行信贷多Agent智能助手 — FastAPI + SSE Streaming",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],

)

# 静态文件服务（前端 HTML/JS/CSS）
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    """根路由返回聊天界面"""
    return FileResponse("static/index.html")


# ==================== API Endpoints ====================

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint — returns SSE streaming response.

    Normal flow: pass message + thread_id
    Resume flow: pass resume dict + thread_id (after GraphInterrupt)
    """
    runtime: AppRuntime = app.state.runtime
    graph = runtime.graph
    if not graph:
        raise HTTPException(status_code=503, detail="Agent graph not initialized")

    thread_id = req.thread_id or str(uuid.uuid4())
    user_id = req.user_id or "anonymous"

    # Build input state
    if req.resume is not None:
        input_state = Command(resume=req.resume)
    else:
        input_state = {"messages": [HumanMessage(content=req.message)]}

    config = {
        ConfigFields.CONFIGURABLE.value: {
            ConfigFields.THREAD_ID.value: thread_id,
        }
    }

    # Set logging context for this request
    set_log_context(user_id=user_id, thread_id=thread_id)

    return StreamingResponse(
        _stream_events(graph, input_state, config, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
        },
    )


@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    """Create a new conversation session, return thread_id."""
    return SessionResponse(thread_id=str(uuid.uuid4()))


@app.get("/api/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str):
    """Get conversation history for the specified session."""
    runtime: AppRuntime = app.state.runtime
    graph = runtime.graph
    if not graph:
        raise HTTPException(status_code=503, detail="Agent graph not initialized")

    config = {
        ConfigFields.CONFIGURABLE.value: {
            ConfigFields.THREAD_ID.value: thread_id,
        }
    }
    try:
        state = await graph.aget_state(config)
    except Exception as e:
        logger.exception("[API] Failed to get state for thread %s", thread_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not state or not state.values:
        return HistoryResponse(thread_id=thread_id, messages=[])

    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content,
            })
    return HistoryResponse(thread_id=thread_id, messages=messages)


@app.get("/api/memory/{user_id}", response_model=MemoryResponse)
async def get_memory(user_id: str):
    """Get user profile memory."""
    runtime: AppRuntime = app.state.runtime
    memory_store = runtime.memory_store
    if not memory_store:
        return MemoryResponse(user_id=user_id, memories=[])

    try:
        memories = memory_store.search(
            user_id=user_id,
            memory_type=MemoryType.USER_PROFILE.value,
            filters={MemoryStatus.STATUS.value: MemoryStatus.ACTIVE.value},
            limit=10,
        )
    except Exception as e:
        logger.exception("[API] Failed to get memory for user %s", user_id)
        return MemoryResponse(user_id=user_id, memories=[])

    return MemoryResponse(
        user_id=user_id,
        memories=[
            {"id": m.id, "content": m.content, "category": m.category}
            for m in memories
        ],
    )


@app.delete("/api/memory/{user_id}")
async def delete_memory(user_id: str):
    """Delete all memories for the specified user."""
    runtime: AppRuntime = app.state.runtime
    memory_store = runtime.memory_store
    if not memory_store:
        return {"status": "no_store"}

    try:
        memory_store.delete_all(user_id=user_id)
    except Exception as e:
        logger.exception("[API] Failed to delete memory for user %s", user_id)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "user_id": user_id}


async def _stream_handoff_resume(graph, resume_cmd: Command, config: dict, thread_id: str,
                                 redis_manager, msg_count_before: int) -> AsyncGenerator[str, None]:
    """
    Stream the graph resume result as SSE.
    After graph completes, extracts new messages and pushes them as SSE 'message' events.
    """
    try:
        await graph.ainvoke(resume_cmd, config=config)

        # Extract new messages generated during resume
        state_snapshot = await graph.aget_state(config)
        all_messages = state_snapshot.values.get("messages", []) if state_snapshot.values else []
        new_messages = all_messages[msg_count_before:]

        for msg in new_messages:
            if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                yield format_sse("message", {"role": role, "content": msg.content})

        yield format_sse("done", {"status": "resolved"})

    except Exception as e:
        logger.exception("[API] Handoff resume streaming failed: thread_id=%s", thread_id)
        yield format_sse("error", {"message": str(e)})

    # Always clean up Redis handoff task
    if redis_manager:
        try:
            client = redis_manager.get_client()
            if client:
                await asyncio.to_thread(client.zrem, "human_handoff:pending", thread_id)
                await asyncio.to_thread(client.delete, f"handoff_task:{thread_id}")
        except Exception as e:
            logger.warning("[API] Failed to clear handoff Redis for %s: %s", thread_id, e)


@app.post("/api/handoff/resolve")
async def resolve_handoff(req: HandoffResolveRequest):
    """
    Human agent resolves HIL handoff — returns SSE stream with the final response.
    action: 'reply' (with content) | 'escalate' | 'close'
    """
    runtime: AppRuntime = app.state.runtime
    graph = runtime.graph
    redis_manager = runtime.redis_manager
    if not graph:
        raise HTTPException(status_code=503, detail="Agent graph not initialized")

    # Build resume command
    resume = {"action": req.action}
    if req.action == "reply" and req.content:
        resume["content"] = req.content

    config = {
        ConfigFields.CONFIGURABLE.value: {
            ConfigFields.THREAD_ID.value: req.thread_id,
        }
    }

    try:
        state_before = await graph.aget_state(config)
        msg_count_before = len(state_before.values.get("messages", [])) if state_before.values else 0
    except Exception:
        msg_count_before = 0

    return StreamingResponse(
        _stream_handoff_resume(graph, Command(resume=resume), config, req.thread_id,
                               redis_manager, msg_count_before),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": req.thread_id,
        },
    )
