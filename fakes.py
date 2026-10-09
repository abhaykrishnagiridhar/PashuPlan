"""Stand-ins for the Anthropic client, so the agent loop is tested without a key or a network."""
import threading
from types import SimpleNamespace as NS


def text(value: str) -> NS:
    return NS(type="text", text=value)


def tool_use(name: str, args: dict | None = None, id: str = "tu_1") -> NS:
    return NS(type="tool_use", id=id, name=name, input=args or {})


def reply(*blocks: NS) -> NS:
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return NS(content=list(blocks), stop_reason=stop)


class ScriptedClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, replies: list[NS]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs) -> NS:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._replies.pop(0)


class FunctionClient:
    """Answers each request with handler(kwargs). Safe to call from several threads."""

    def __init__(self, handler):
        self._handler = handler
        self._lock = threading.Lock()
        self.calls: list[dict] = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs) -> NS:
        with self._lock:
            self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._handler(kwargs)
