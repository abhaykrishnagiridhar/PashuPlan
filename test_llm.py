"""The OpenAI-compatible adapter (used for Gemini, Groq, OpenRouter, Ollama), tested without any network."""
import json

import pytest

from pashuplan import agents
from pashuplan.llm import OpenAICompatClient

BASE = "https://example.test/v1"
KEY = "secret-key-123"


class FakeHttp:
    """Replaces the HTTP call. Each scripted item is (status, headers, body)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        return self.replies.pop(0)


def ok(message, finish="stop"):
    return 200, {}, {"choices": [{"message": message, "finish_reason": finish}]}


def text_reply(content):
    return ok({"role": "assistant", "content": content})


def tool_reply(*calls, content=None):
    return ok({"role": "assistant", "content": content, "tool_calls": list(calls)}, finish="tool_calls")


def call(name, args="{}", id="call_1", **extra):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}, **extra}


def make(http, **kw):
    kw.setdefault("sleep", lambda s: None)
    return OpenAICompatClient(BASE, KEY, http=http, **kw)


TOOLS = [{"name": "get_x", "description": "Gets x.",
          "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []}}]


def test_request_shape_url_auth_model_system_and_tools():
    http = FakeHttp(text_reply("hi"))
    make(http).messages.create(model="m1", max_tokens=300, system="You are X.",
                               messages=[{"role": "user", "content": "Question?"}], tools=TOOLS)
    sent = http.calls[0]
    assert sent["url"] == BASE + "/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    body = sent["body"]
    assert body["model"] == "m1" and body["max_tokens"] == 300
    assert body["messages"][0] == {"role": "system", "content": "You are X."}
    assert body["messages"][1] == {"role": "user", "content": "Question?"}
    assert body["tools"] == [{"type": "function", "function": {
        "name": "get_x", "description": "Gets x.", "parameters": TOOLS[0]["input_schema"]}}]
    assert body["tool_choice"] == "auto"


def test_no_tools_key_when_tools_are_not_offered():
    http = FakeHttp(text_reply("hi"))
    make(http).messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert "tools" not in http.calls[0]["body"] and "tool_choice" not in http.calls[0]["body"]


def test_trailing_slash_on_the_base_url_is_fine():
    http = FakeHttp(text_reply("hi"))
    OpenAICompatClient(BASE + "/", KEY, http=http).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert http.calls[0]["url"] == BASE + "/chat/completions"


def test_minimum_token_budget_and_extra_body_are_applied():
    http = FakeHttp(text_reply("hi"))
    make(http, min_max_tokens=4096, extra_body={"reasoning_effort": "low"}).messages.create(
        model="m", max_tokens=1024, system="s", messages=[{"role": "user", "content": "q"}])
    assert http.calls[0]["body"]["max_tokens"] == 4096
    assert http.calls[0]["body"]["reasoning_effort"] == "low"


def test_plain_text_answer_becomes_a_text_block():
    reply = make(FakeHttp(text_reply("The answer."))).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert [(b.type, b.text) for b in reply.content] == [("text", "The answer.")]
    assert reply.stop_reason == "end_turn"


def test_tool_call_becomes_a_tool_use_block_with_parsed_input():
    reply = make(FakeHttp(tool_reply(call("get_x", '{"n": 3}', id="abc")))).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}], tools=TOOLS)
    block = reply.content[0]
    assert (block.type, block.id, block.name, block.input) == ("tool_use", "abc", "get_x", {"n": 3})
    assert reply.stop_reason == "tool_use"


def test_invalid_tool_arguments_are_reported_not_crashed_on():
    reply = make(FakeHttp(tool_reply(call("get_x", "{oops")))).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}], tools=TOOLS)
    assert reply.content[0].input == {"_invalid_arguments": "{oops"}


def test_a_missing_tool_call_id_is_replaced_with_a_unique_one():
    c = {"type": "function", "function": {"name": "get_x", "arguments": "{}"}}
    reply = make(FakeHttp(tool_reply(c, dict(c)))).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}], tools=TOOLS)
    ids = [b.id for b in reply.content]
    assert all(ids) and len(set(ids)) == 2


def test_a_cut_off_answer_reports_max_tokens():
    reply = make(FakeHttp(ok({"role": "assistant", "content": "partial"}, finish="length"))).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert reply.stop_reason == "max_tokens"


def test_anthropic_style_history_is_translated_for_the_next_request():
    http = FakeHttp(text_reply("done"))
    history = [
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                                          {"type": "tool_use", "id": "t1", "name": "get_x", "input": {"n": 2}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"v": 1}'},
                                     {"type": "text", "text": "Answer now."}]},
    ]
    make(http).messages.create(model="m", max_tokens=10, system="s", messages=history)
    sent = http.calls[0]["body"]["messages"]
    assert sent[2] == {"role": "assistant", "content": "Checking.", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "get_x", "arguments": '{"n": 2}'}}]}
    assert sent[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"v": 1}'}
    assert sent[4] == {"role": "user", "content": "Answer now."}


def test_an_assistant_turn_with_only_tool_calls_has_no_text():
    http = FakeHttp(text_reply("done"))
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_x", "input": {}}]}]
    make(http).messages.create(model="m", max_tokens=10, system="s", messages=history)
    assert http.calls[0]["body"]["messages"][2]["content"] is None


def test_extra_fields_on_a_tool_call_such_as_thought_signatures_are_echoed_back():
    signature = {"google": {"thought_signature": "sig-xyz"}}
    http = FakeHttp(tool_reply(call("get_x", id="t9", extra_content=signature)), text_reply("done"))
    client = make(http)
    first = client.messages.create(model="m", max_tokens=10, system="s",
                                   messages=[{"role": "user", "content": "q"}], tools=TOOLS)
    use = first.content[0]
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": [{"type": "tool_use", "id": use.id, "name": use.name, "input": use.input}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "{}"}]}]
    client.messages.create(model="m", max_tokens=10, system="s", messages=history, tools=TOOLS)
    echoed = http.calls[1]["body"]["messages"][2]["tool_calls"][0]
    assert echoed["extra_content"] == signature


def test_rate_limits_are_retried_after_the_advised_wait():
    waits = []
    http = FakeHttp((429, {"retry-after": "7"}, {"error": {"message": "slow down"}}), text_reply("finally"))
    reply = OpenAICompatClient(BASE, KEY, http=http, sleep=waits.append).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert reply.content[0].text == "finally"
    assert waits == [7.0] and len(http.calls) == 2


def test_server_errors_back_off_and_a_hopeless_rate_limit_finally_raises():
    waits = []
    busy = (503, {}, {"error": {"message": "overloaded"}})
    http = FakeHttp(*[busy] * 3)
    client = OpenAICompatClient(BASE, KEY, http=http, sleep=waits.append, max_retries=2)
    with pytest.raises(RuntimeError, match="503"):
        client.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert len(http.calls) == 3 and len(waits) == 2 and waits[1] > waits[0]


def google_429(retry_delay=None, quota_id=None, wrap_in_list=True):
    """Shaped like the real Google rate-limit reply, which arrives as a JSON list."""
    details = []
    if quota_id:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaMetric": "generativelanguage.googleapis.com/x", "quotaId": quota_id}]})
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    error = {"error": {"code": 429, "message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED",
                       "details": details}}
    return 429, {}, [error] if wrap_in_list else error


def test_the_retry_delay_google_puts_in_the_body_is_honoured():
    waits = []
    http = FakeHttp(google_429(retry_delay="34s"), google_429(retry_delay="12.5s", wrap_in_list=False), text_reply("ok"))
    reply = OpenAICompatClient(BASE, KEY, http=http, sleep=waits.append).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert reply.content[0].text == "ok"
    assert waits == [34.0, 12.5]


def test_a_per_minute_limit_is_retried_but_a_daily_limit_is_not():
    waits = []
    daily = google_429(quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    http = FakeHttp(daily)
    with pytest.raises(RuntimeError, match="daily") as err:
        OpenAICompatClient(BASE, KEY, http=http, sleep=waits.append).messages.create(
            model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert len(http.calls) == 1 and waits == []
    assert "GenerateRequestsPerDayPerProjectPerModel" in str(err.value)
    minute = google_429(retry_delay="5s", quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    http2 = FakeHttp(minute, text_reply("ok"))
    OpenAICompatClient(BASE, KEY, http=http2, sleep=waits.append).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert waits == [5.0]


def test_the_final_error_names_the_quota_and_is_not_cut_short():
    http = FakeHttp(*[google_429(retry_delay="1s", quota_id="GenerateContentInputTokensPerModelPerMinute-FreeTier")] * 2)
    client = OpenAICompatClient(BASE, KEY, http=http, sleep=lambda s: None, max_retries=1)
    with pytest.raises(RuntimeError) as err:
        client.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    text = str(err.value)
    assert "429" in text and "GenerateContentInputTokensPerModelPerMinute-FreeTier" in text
    assert "exceeded your current quota" in text


def test_auth_errors_are_not_retried_and_never_leak_the_key():
    http = FakeHttp((401, {}, {"error": {"message": f"bad key {KEY}"}}))
    with pytest.raises(RuntimeError) as err:
        make(http).messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert "401" in str(err.value) and KEY not in str(err.value) and len(http.calls) == 1


def test_an_empty_or_odd_response_is_an_error_not_a_silent_blank():
    for bad in [(200, {}, {"choices": []}), (200, {}, {"unexpected": True}), (200, {}, "not json")]:
        with pytest.raises(RuntimeError, match="response"):
            make(FakeHttp(bad)).messages.create(model="m", max_tokens=10, system="s",
                                                messages=[{"role": "user", "content": "q"}])


@pytest.fixture
def local_server():
    """A real HTTP server on localhost, so the transport is tested over an actual socket."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []
    script = {"status": 200, "headers": {}, "body": b"{}"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                         "body": json.loads(self.rfile.read(length) or b"{}")})
            self.send_response(script["status"])
            for k, v in script["headers"].items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(script["body"])

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen, script
    server.shutdown()


def test_the_real_transport_sends_json_with_the_key_and_reads_the_answer(local_server):
    base, seen, script = local_server
    script["body"] = json.dumps({"choices": [{"message": {"role": "assistant", "content": "over a socket"},
                                              "finish_reason": "stop"}]}).encode()
    reply = OpenAICompatClient(base, KEY).messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert reply.content[0].text == "over a socket"
    assert seen[0]["path"] == "/v1/chat/completions" and seen[0]["auth"] == f"Bearer {KEY}"
    assert seen[0]["body"]["messages"][0]["role"] == "system"


def test_the_real_transport_reads_error_bodies_and_retry_after_headers(local_server):
    base, seen, script = local_server
    waits = []
    script.update(status=429, headers={"Retry-After": "3"}, body=b'{"error": {"message": "quota"}}')
    client = OpenAICompatClient(base, KEY, sleep=waits.append, max_retries=1)
    with pytest.raises(RuntimeError, match="429.*quota"):
        client.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert waits == [3.0] and len(seen) == 2


def test_the_real_transport_reports_a_server_that_is_not_there():
    client = OpenAICompatClient("http://127.0.0.1:9/v1", KEY, max_retries=0, timeout=2)
    with pytest.raises(RuntimeError, match="could not reach"):
        client.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])


def test_the_agent_loop_runs_end_to_end_through_the_adapter(farm):
    http = FakeHttp(tool_reply(call("get_mastitis_suspects", id="c1")), text_reply("C16, C32 and C40 are sick."))
    res = agents.run_agent(make(http), farm, "health", "Why is my milk down?", "en", model="gem-test")
    assert res.answer == "C16, C32 and C40 are sick." and res.turns == 2
    tool_msg = http.calls[1]["body"]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1" and "C16" in tool_msg["content"]
    assert json.loads(tool_msg["content"])["suspects"]
