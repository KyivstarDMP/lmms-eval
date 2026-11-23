from __future__ import annotations

import asyncio
import json
import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import openai

from lmms_eval.models import get_model
from lmms_eval.models.chat import openai_stream
from lmms_eval.models.chat.openai_stream import OpenAIStreamChat

try:
    import httpx2 as http  # what openai>=3 runs on
except ImportError:
    import httpx as http

STALL_S = 0.4
FAILED = "[LMMS_EVAL_REQUEST_FAILED"


def _chunk(content: str | None = None, *, usage: dict | None = None) -> bytes:
    body = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": []}
    if content is not None:
        body["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n".encode()


def _sse(*pieces: str, every: float = 0.0, first_after: float = 0.0, hang_after: int | None = None, usage: dict | None = None):
    async def body():
        await asyncio.sleep(first_after)
        for index, piece in enumerate(pieces):
            if index == hang_after:
                await asyncio.sleep(3600)
            yield _chunk(piece)
            await asyncio.sleep(every)
        if usage is not None:
            yield _chunk(usage=usage)
        yield b"data: [DONE]\n\n"

    return http.Response(200, headers={"content-type": "text/event-stream"}, content=body())


def _usage(prompt: int, completion: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _stream(*chunks: bytes):
    async def body():
        for chunk in chunks:
            yield chunk
        yield b"data: [DONE]\n\n"

    return http.Response(200, headers={"content-type": "text/event-stream"}, content=body())


def _question(request) -> str:
    return json.loads(request.content)["messages"][0]["content"][0]["text"]


def _requests(count: int, **gen_kwargs) -> list[SimpleNamespace]:
    def doc_to_messages(doc):
        return [{"role": "user", "content": [{"type": "text", "text": doc["q"]}]}]

    args = ("", doc_to_messages, {"max_new_tokens": 16, **gen_kwargs})
    return [SimpleNamespace(args=(*args, index, "demo", "test")) for index in range(count)]


def _model(handler, **model_args) -> OpenAIStreamChat:
    model = OpenAIStreamChat(model="m", base_url="http://server/v1", timeout=STALL_S, max_retries=3, retry_backoff_s=0, num_concurrent=4, **model_args)
    model.task_dict = {"demo": {"test": [{"q": f"q{index}"} for index in range(8)]}}
    model._make_client = lambda: openai.AsyncOpenAI(
        base_url="http://server/v1",
        api_key="EMPTY",
        max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=http.MockTransport(handler)),
    )
    return model


class TestOpenAIStreamAnswers(unittest.TestCase):
    def test_answers_are_folded_from_the_stream_in_request_order(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _sse(_question(request).upper(), "!", usage={"prompt_tokens": 5, "completion_tokens": 2})

        model = _model(handler, chat_template_kwargs='{"enable_thinking":false}')
        results = model.generate_until(_requests(3, temperature=0, top_k=20, reasoning_effort="low", top_p=None))

        self.assertEqual([result.text for result in results], ["Q0!", "Q1!", "Q2!"])
        self.assertEqual((results[0].token_counts.input_tokens, results[0].token_counts.output_tokens), (5, 2))
        for body in bodies:
            self.assertTrue(body["stream"])
            self.assertEqual((body["max_tokens"], body["temperature"], body["top_k"], body["n"]), (16, 0, 20, 1))
            for absent in ("top_p", "reasoning_effort", "stream_options", "truncate_prompt_tokens"):
                self.assertNotIn(absent, body)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_a_context_budget_truncates_the_prompt_and_ends_the_answer_where_a_server_at_the_budget_would(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            if _question(request) == "q0":
                return _stream(_chunk("a", usage=_usage(60, 1)), _chunk("b", usage=_usage(60, 4)), _chunk("c", usage=_usage(60, 5)))
            return _stream(_chunk("x", usage=_usage(64, 1)))

        filled, over = _model(handler, max_model_length=64).generate_until(_requests(2))

        self.assertEqual(filled.text, "ab")
        self.assertTrue(over.text.startswith(f"{FAILED} after 1 retries] over_budget: a 64-token prompt reaches the 64-token context budget"))
        self.assertEqual(bodies[0]["truncate_prompt_tokens"], 48)
        self.assertEqual(bodies[0]["stream_options"], {"include_usage": True, "continuous_usage_stats": True})

    def test_the_temperature_is_always_sent(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _sse("ok")

        _model(handler).generate_until(_requests(1))

        self.assertEqual(bodies[0]["temperature"], 0)

    def test_parsed_reasoning_is_not_part_of_the_answer(self):
        def handler(request):
            reasoning = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": {"reasoning": "thinking"}, "finish_reason": None}]}
            return _stream(f"data: {json.dumps(reasoning)}\n\n".encode(), _chunk("42"))

        self.assertEqual(_model(handler).generate_until(_requests(1))[0].text, "42")


class TestOpenAIStreamStalls(unittest.TestCase):
    def test_a_queued_request_outlives_the_stall_bound_while_others_stream(self):
        async def handler(request):
            question = _question(request)
            if question == "q0":
                return _sse(*["x"] * 12, every=0.1)
            if question == "q1":
                await asyncio.sleep(1.0)  # no response headers yet
            return _sse("late", first_after=1.0 if question == "q2" else 0.0)

        results = _model(handler).generate_until(_requests(3))

        self.assertEqual([result.text for result in results], ["x" * 12, "late", "late"])

    def test_a_started_stream_that_goes_quiet_fails_without_a_retry(self):
        calls = Counter()

        def handler(request):
            question = _question(request)
            calls[question] += 1
            return _sse("a", "b", hang_after=1) if question == "q0" else _sse("ok")

        results = _model(handler).generate_until(_requests(2))

        self.assertTrue(results[0].text.startswith(f"{FAILED} after 1 retries] stalled: _StreamStall: no chunk from this stream"))
        self.assertEqual(results[1].text, "ok")
        self.assertEqual(calls["q0"], 1)

    def test_consecutive_stalls_abort_the_evaluation(self):
        def handler(request):
            return _sse("never", first_after=3600)

        with mock.patch.object(openai_stream, "_STALL_ABORT_AFTER", 2):
            with self.assertRaisesRegex(RuntimeError, "stalled on 2 consecutive requests"):
                _model(handler).generate_until(_requests(3))


class TestOpenAIStreamRetries(unittest.TestCase):
    def test_server_errors_back_off_and_bad_requests_fail_at_once(self):
        calls = Counter()

        def handler(request):
            question = _question(request)
            calls[question] += 1
            if question == "q0" and calls[question] == 1:
                return http.Response(503, json={"error": {"message": "overloaded"}})
            if question == "q1":
                return http.Response(400, json={"error": {"message": "prompt too long"}})
            if question == "q2":
                return http.Response(500, json={"error": {"message": "boom"}})
            return _sse("ok")

        results = _model(handler).generate_until(_requests(3))

        self.assertEqual((results[0].text, calls["q0"]), ("ok", 2))
        self.assertTrue(results[1].text.startswith(f"{FAILED} after 1 retries] rejected: BadRequestError: Error code: 400"))
        self.assertEqual(calls["q1"], 1)
        self.assertTrue(results[2].text.startswith(f"{FAILED} after 6 retries] retries_exhausted: InternalServerError: Error code: 500"))
        self.assertEqual(calls["q2"], 6)

    def test_connection_retries_leave_the_backoff_attempts_alone(self):
        calls = Counter()

        def handler(request):
            calls["q0"] += 1
            if calls["q0"] <= 2:
                raise http.ConnectError("connection reset", request=request)
            if calls["q0"] <= 4:
                return http.Response(503, json={"error": {"message": "overloaded"}})
            return _sse("ok")

        with mock.patch.object(openai_stream, "_CONNECT_RETRY_SLEEP_S", 0):
            results = _model(handler).generate_until(_requests(1))

        self.assertEqual(results[0].text, "ok")
        self.assertEqual(calls["q0"], 5)

    def test_an_unreachable_server_aborts_the_evaluation(self):
        def handler(request):
            raise http.ConnectError("connection refused", request=request)

        with mock.patch.object(openai_stream, "_CONNECT_RETRY_SLEEP_S", 0):
            with self.assertRaisesRegex(RuntimeError, "unreachable"):
                _model(handler).generate_until(_requests(1))


class TestOpenAIStreamClient(unittest.TestCase):
    def test_the_client_leaves_retries_to_the_backend(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            client = OpenAIStreamChat(model="m", base_url="http://server/v1", timeout=30, num_concurrent=3)._make_client()

        self.assertEqual((str(client.base_url), client.api_key), ("http://server/v1/", "EMPTY"))
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 0)

    def test_the_limit_sheds_on_a_full_kv_cache_and_grows_only_while_the_server_keeps_up(self):
        load = openai_stream._parse_server_load("vllm:num_requests_running 60\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.5\nvllm:num_preemptions_total 2\n")

        self.assertEqual(openai_stream._next_limit(64, ceiling=256, in_flight=64, load=load, preempted=0), 96)
        self.assertEqual(openai_stream._next_limit(64, ceiling=256, in_flight=64, load=load._replace(kv_usage=0.97), preempted=0), 54)
        self.assertEqual(openai_stream._next_limit(64, ceiling=256, in_flight=64, load=load._replace(waiting=5), preempted=0), 64)
        self.assertIsNone(openai_stream._parse_server_load("vllm:num_requests_running 60\n"))

    def test_the_registry_resolves_openai_stream(self):
        self.assertIs(get_model("openai_stream"), OpenAIStreamChat)


if __name__ == "__main__":
    unittest.main()
