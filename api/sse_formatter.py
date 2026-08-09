"""
SSE (Server-Sent Events) frame formatter.
Converts event type + data dict to standard SSE text frame.
"""
import json


def format_sse(event: str, data: dict) -> str:
    """
    Format an event as an SSE frame.

    Output format:
        event: <type>\\n
        data: <json>\\n
        \\n

    Args:
        event: SSE event type (token / tool_start / tool_end / handoff / done / error)
        data: Event payload dict (serialized to JSON)

    Returns:
        SSE-formatted string with trailing double newline
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
