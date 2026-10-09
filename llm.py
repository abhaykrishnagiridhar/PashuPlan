"""A small client for OpenAI-compatible chat endpoints (Google Gemini, Groq, OpenRouter, Ollama).

It exposes the same `client.messages.create(...)` call the agent loop already uses with the
Anthropic SDK, and translates Anthropic-style messages and tool blocks to and from the OpenAI
chat format, so the agents do not change when the provider does.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace
from urllib.parse import urlparse

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_WAIT_SECONDS = 90.0
MAX_BACKOFF_SECONDS = 30.0
ERROR_TEXT_LIMIT = 600
DAILY_QUOTA_HINT = ("The daily free quota is used up. Try again tomorrow, or set PASHUPLAN_MODEL to a different "
                    "model, which has its own quota.")


def _error_info(payload: object) -> tuple[str, float | None, list[str]]:
    """(message, seconds the server asks us to wait, quota ids hit) from an error body.

    Google sends errors as a JSON list holding one {"error": {...}} object, with RetryInfo and QuotaFailure details.
    """
    body = payload[0] if isinstance(payload, list) and payload else payload
    error = body.get("error", body) if isinstance(body, dict) else body
    if not isinstance(error, dict):
        return str(error), None, []
    retry, quotas = None, []
    for detail in error.get("details") or []:
        kind = str(detail.get("@type", ""))
        if kind.endswith("RetryInfo"):
            try:
                retry = float(str(detail.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                pass
        elif kind.endswith("QuotaFailure"):
            quotas += [v["quotaId"] for v in detail.get("violations", []) if v.get("quotaId")]
    return str(error.get("message", error)), retry, quotas


def _post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict, object]:
    """POST JSON with the standard library. Returns (status, lower-cased headers, parsed JSON or text)."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            status, reply_headers, raw = reply.status, reply.headers, reply.read()
    except urllib.error.HTTPError as err:  # 4xx and 5xx still carry a useful body
        status, reply_headers, raw = err.code, err.headers, err.read()
    except urllib.error.URLError as err:
        raise RuntimeError(f"could not reach {urlparse(url).netloc}: {err.reason}") from err
    text = raw.decode("utf-8", errors="replace")
    try:
        payload: object = json.loads(text)
    except ValueError:
        payload = text
    return status, {k.lower(): v for k, v in reply_headers.items()}, payload


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]


def to_openai_messages(system: str, messages: list[dict], extras: dict[str, dict]) -> list[dict]:
    """Anthropic-style history to OpenAI chat messages. `extras` maps tool call id to extra fields to echo back."""
    out: list[dict] = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            texts = [b["text"] for b in content if b["type"] == "text"]
            calls = []
            for b in content:
                if b["type"] != "tool_use":
                    continue
                call = {"id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                if b["id"] in extras:
                    call["extra_content"] = extras[b["id"]]
                calls.append(call)
            assistant = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                assistant["tool_calls"] = calls
            out.append(assistant)
        else:
            texts = []
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
                elif b["type"] == "text":
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str = "", *, http=_post, sleep=time.sleep, max_retries: int = 4,
                 min_max_tokens: int = 0, extra_body: dict | None = None, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.min_max_tokens = min_max_tokens
        self.extra_body = dict(extra_body or {})
        self._key = api_key
        self._http, self._sleep = http, sleep
        self._max_retries, self._timeout = max_retries, timeout
        self._extras: dict[str, dict] = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, max_tokens: int, system: str, messages: list[dict], tools: list[dict] | None = None):
        body = {"model": model, "max_tokens": max(max_tokens, self.min_max_tokens), **self.extra_body,
                "messages": to_openai_messages(system, messages, self._extras)}
        if tools:
            body["tools"], body["tool_choice"] = to_openai_tools(tools), "auto"
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return self._parse(self._send(self.base_url + "/chat/completions", headers, body))

    def _send(self, url: str, headers: dict, body: dict) -> object:
        for attempt in range(self._max_retries + 1):
            status, reply_headers, payload = self._http(url, headers, body, self._timeout)
            if status == 200:
                return payload
            message, server_delay, quotas = _error_info(payload)
            daily = any("PerDay" in q for q in quotas)  # waiting a minute will not fix a daily limit
            if status in RETRY_STATUSES and not daily and attempt < self._max_retries:
                self._sleep(self._wait_seconds(reply_headers, server_delay, attempt))
                continue
            raise RuntimeError(self._error_text(url, status, message, quotas, daily))
        raise AssertionError("unreachable")

    @staticmethod
    def _wait_seconds(headers: dict, server_delay: float | None, attempt: int) -> float:
        try:
            return min(float(headers["retry-after"]), MAX_WAIT_SECONDS)
        except (KeyError, ValueError):
            pass
        if server_delay is not None:
            return min(server_delay, MAX_WAIT_SECONDS)
        return min(float(2 ** (attempt + 1)), MAX_BACKOFF_SECONDS)

    def _error_text(self, url: str, status: int, message: str, quotas: list[str], daily: bool) -> str:
        text = f"{status} from {urlparse(url).netloc}: {message[:ERROR_TEXT_LIMIT]}"
        if quotas:
            text += f" (quota: {', '.join(quotas)})"
        if daily:
            text += f" {DAILY_QUOTA_HINT}"
        return text.replace(self._key, "***") if self._key else text

    def _parse(self, payload: object):
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise RuntimeError("unexpected response from the model server (no choices)")
        message, finish = choices[0].get("message") or {}, choices[0].get("finish_reason")
        blocks = []
        if message.get("content"):
            blocks.append(SimpleNamespace(type="text", text=message["content"]))
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {"_invalid_arguments": raw}
            if call.get("extra_content"):
                self._extras[call_id] = call["extra_content"]
            blocks.append(SimpleNamespace(type="tool_use", id=call_id, name=call["function"]["name"], input=args))
        stop = "tool_use" if tool_calls else ("max_tokens" if finish == "length" else "end_turn")
        return SimpleNamespace(content=blocks, stop_reason=stop)
