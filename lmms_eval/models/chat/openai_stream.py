import asyncio
import contextlib
import json
import os
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, NamedTuple

import openai
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.concurrency_control import parse_bool
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.usage_metrics import log_usage
from lmms_eval.protocol import ChatMessages

try:
    import httpx2 as httpx  # what openai>=3 runs on
except ImportError:
    import httpx

# The request logic of lighteval's `vllm-openai` client (KyivstarDMP/lighteval, vllm_openai_model.py), chat
# generation route. Keep the two in step: a model must be served to both evaluators under the same rules.

_STANDARD_PARAMS = ("top_p", "seed", "frequency_penalty", "presence_penalty")
_EXTRA_BODY_PARAMS = {"top_k": "top_k", "min_p": "min_p", "repetition_penalty": "repetition_penalty", "skip_special_tokens": "skip_special_tokens", "min_new_tokens": "min_tokens"}

_CONNECT_RETRIES = 2
_CONNECT_RETRY_SLEEP_S = 1.0
_STALL_ABORT_AFTER = 8  # consecutive stalled requests before the server is declared dead
_DEFAULT_STALL_S = 600.0
_MAX_BACKOFF_S = 64.0
_KEEPALIVE_EXPIRY_S = 2.0  # below vLLM's HTTP keep-alive (5 s): never reuse a connection the server closed

_LOAD_INTERVAL_S = 2.0
_START_CONCURRENCY = 64
_KV_GROW_BELOW = 0.85  # grow only while the KV cache has room; between this and _KV_SHED_AT, hold
_KV_SHED_AT = 0.95
_SHED_TO = 0.9  # shed to this fraction of what the server is running
_SATURATED = 0.9  # "all" with slack: the client fills its limit, the server runs what is in flight


class _ProgressClock:
    """When any stream last received a chunk: a request still queued behind others stalls only if this went quiet."""

    def __init__(self) -> None:
        self.last = time.monotonic()

    def tick(self) -> None:
        self.last = time.monotonic()


class _Quiet(Exception):
    pass


class _StreamStall(openai.APITimeoutError):
    def __init__(self, detail: str, request: httpx.Request) -> None:
        openai.APIConnectionError.__init__(self, message=detail, request=request)


class _OverBudget(Exception):
    """A prompt that a server started at the context budget would reject."""


class _Failed(NamedTuple):
    reason: str  # stalled | rejected | retries_exhausted | over_budget
    detail: str
    attempts: int


class _Answer(NamedTuple):
    text: str
    usage: Any


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _hold_prompt(usage: Any, budget: int | None) -> None:
    if budget is not None and usage is not None and usage.prompt_tokens >= budget:
        raise _OverBudget(f"a {usage.prompt_tokens}-token prompt reaches the {budget}-token context budget")


async def _close(stream: Any) -> None:
    closer = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if closer is not None:
        with contextlib.suppress(Exception):
            await closer()


async def _await_unless_quiet(awaitable: Awaitable[Any], *, quiet_for: Callable[[], float], stall_s: float) -> Any:
    """Await ``awaitable``, raising ``_Quiet`` once ``quiet_for()`` reaches ``stall_s`` seconds before it finishes."""
    task = asyncio.ensure_future(awaitable)
    poll = min(15.0, stall_s / 4)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll)
            if done:
                return task.result()
            if quiet_for() >= stall_s:
                raise _Quiet
    finally:
        if not task.done():
            task.cancel()


async def _stream_chunks(stream: Any, *, progress: _ProgressClock, stall_s: float) -> AsyncIterator[Any]:
    """Yield ``stream``'s chunks, raising ``_StreamStall`` when it stalls.

    Before its first chunk a request is queued behind others, not stuck: it stalls only when no stream anywhere has
    received a chunk for ``stall_s``. Once started, the gap between its own chunks is bounded by ``stall_s``.
    """
    iterator = stream.__aiter__()
    opened = last_chunk = time.monotonic()
    started = False

    def quiet_for() -> float:
        return time.monotonic() - (last_chunk if started else max(progress.last, opened))

    try:
        while True:
            try:
                chunk = await _await_unless_quiet(iterator.__anext__(), quiet_for=quiet_for, stall_s=stall_s)
            except StopAsyncIteration:
                return
            except _Quiet:
                source = "this stream" if started else "any stream"
                request = getattr(getattr(stream, "response", None), "request", None)
                raise _StreamStall(f"no chunk from {source} for {stall_s:g}s", request or httpx.Request("POST", "http://vllm/stream")) from None
            started = True
            last_chunk = time.monotonic()
            progress.tick()
            yield chunk
    finally:
        await _close(stream)


async def _collect(stream: Any, *, progress: _ProgressClock, stall_s: float, budget: int | None) -> _Answer:
    """Fold a chat-completion stream into its answer, held to the context ``budget``.

    A server window wider than the budget must not change the result, so the stream ends as one started at the
    budget would end it: a prompt at or over the budget raises ``_OverBudget``, and the answer stops on the chunk
    whose continuous usage reaches the budget. Parsed reasoning (``delta.reasoning``) is not part of the answer.
    """
    pieces: list[str] = []
    last_usage = None
    chunks = _stream_chunks(stream, progress=progress, stall_s=stall_s)
    try:
        async for chunk in chunks:
            usage = getattr(chunk, "usage", None)
            last_usage = usage or last_usage
            _hold_prompt(usage, budget)
            choices = [choice for choice in getattr(chunk, "choices", None) or [] if not (getattr(choice, "index", 0) or 0)]
            for choice in choices:
                delta = getattr(choice, "delta", None)
                if delta is not None and (piece := getattr(delta, "content", None)):
                    pieces.append(piece)
            if choices and budget is not None and usage is not None and usage.prompt_tokens + (usage.completion_tokens or 0) >= budget:
                break
    finally:
        await _close(chunks)
    return _Answer("".join(pieces), last_usage)


class _ServerLoad(NamedTuple):
    running: float
    waiting: float
    kv_usage: float
    preemptions: float  # cumulative


_LOAD_SERIES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:num_preemptions_total": "preemptions",
}
_SERIES_LINE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def _metrics_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/metrics"


def _parse_server_load(text: str) -> _ServerLoad | None:
    """Read the load series from a Prometheus exposition; ``None`` if the server lacks any of them.

    Several engines (data parallel) add up, except the KV usage, where the fullest one counts.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _SERIES_LINE.match(line)
        if match is None or (field := _LOAD_SERIES.get(match["name"])) is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        totals[field] = max(totals.get(field, 0.0), value) if field == "kv_usage" else totals.get(field, 0.0) + value
    return _ServerLoad(**totals) if len(totals) == len(_LOAD_SERIES) else None


def _next_limit(limit: int, *, ceiling: int, in_flight: int, load: _ServerLoad, preempted: float) -> int:
    """One control step.

    Shed below what the server is running once its KV cache is exhausted or it just preempted, by at most half, and
    not again until the last shed has taken effect. Grow only while the client fills its limit and the server is
    running what the client has in flight, with KV to spare. Otherwise hold.
    """
    if in_flight > limit:
        return limit
    if load.kv_usage >= _KV_SHED_AT or preempted > 0:
        return max(1, min(limit, max(int(load.running * _SHED_TO), limit // 2)))
    saturating = in_flight >= _SATURATED * limit
    seen = load.waiting == 0 and load.running >= _SATURATED * in_flight
    if saturating and seen and load.kv_usage < _KV_GROW_BELOW:
        return min(ceiling, limit + max(1, limit // 2))
    return limit


class _AdaptiveLimiter:
    """An in-flight gate with a movable limit, used like a semaphore; lowering the limit evicts nothing."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight = 0
        self._waiters: deque[asyncio.Future] = deque()

    async def __aenter__(self) -> None:
        if self.in_flight < self.limit and not self._waiters:
            self.in_flight += 1
            return
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter  # `_wake` counts the slot before resolving it
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                self.in_flight -= 1  # granted a slot but cancelled before using it
                self._wake()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
            raise

    async def __aexit__(self, *exc_info: object) -> None:
        self.in_flight -= 1
        self._wake()

    def set_limit(self, limit: int) -> None:
        self.limit = limit
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.in_flight < self.limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.in_flight += 1
                waiter.set_result(None)


async def _steer(limiter: _AdaptiveLimiter, *, ceiling: int, fetch: Callable[[], Awaitable[str | None]], interval: float = _LOAD_INTERVAL_S) -> None:
    """Move ``limiter`` with the server's load until cancelled.

    A server that has never reported its load is driven at the ceiling; one that did and now misses a reading keeps
    the current limit.
    """
    last_preemptions: float | None = None
    warned = False
    while True:
        await asyncio.sleep(interval)
        text = await fetch()
        load = _parse_server_load(text) if text else None
        if load is None:
            if last_preemptions is not None:
                continue
            if not warned:
                eval_logger.warning(f"Adaptive concurrency: the server reports no load; sending up to {ceiling} in flight.")
                warned = True
            limiter.set_limit(ceiling)
            continue
        preempted = 0.0 if last_preemptions is None else max(0.0, load.preemptions - last_preemptions)
        last_preemptions = load.preemptions
        limit = _next_limit(limiter.limit, ceiling=ceiling, in_flight=limiter.in_flight, load=load, preempted=preempted)
        if limit != limiter.limit:
            eval_logger.info(f"Adaptive concurrency {limiter.limit} -> {limit} (in flight={limiter.in_flight}; server running={load.running:.0f} waiting={load.waiting:.0f} kv={load.kv_usage:.0%} preempted={preempted:.0f})")
            limiter.set_limit(limit)


def _fit_open_files(connections: int) -> None:
    """Raise the soft open-files limit to fit ``connections`` sockets: containers commonly start it at 1024."""
    try:
        import resource
    except ImportError:  # no such limit on Windows
        return
    wanted = connections + 1024  # headroom for the process's own files
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= wanted:
        return
    target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    if target < wanted:
        eval_logger.warning(f"The open-files limit ({hard}) cannot fit {connections} connections: requests past ~{target - 1024} in flight will fail to connect")


@register_model("openai_stream")
class OpenAIStreamChat(lmms):
    """Streaming chat client for a vLLM server, with the request rules of lighteval's ``vllm-openai`` client.

    - ``timeout`` is a stall bound (600 s when unset), never the length of a generation: a started request stalls
      when its own chunks stop for that long, a queued one only when no stream at all has received a chunk for that
      long. A stall is not retried; 8 consecutive stalls abort the evaluation.
    - A connection failure is retried twice, then aborts the evaluation. HTTP 429, 5xx and an error reported inside
      a stream back off (``retry_backoff_s`` times ``retry_multiplier`` per attempt, capped at 64 s). Any other 4xx
      fails that request only.
    - ``max_model_length`` is the context budget held per request whatever window the server has: a text-only
      prompt is left-truncated by the server to leave room for the answer, a prompt at or over the budget fails,
      and a generation stops once it fills the budget.
    - ``adaptive_concurrency`` makes ``num_concurrent`` a ceiling and follows the server's ``/metrics`` load.

    A request the client gives up on is answered ``[LMMS_EVAL_REQUEST_FAILED after N retries] <reason>: <error>``,
    the reason being ``stalled``, ``rejected``, ``retries_exhausted`` or ``over_budget``.
    """

    is_simple = False

    def __init__(
        self,
        model: str | None = None,
        model_version: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = 5,
        retry_backoff_s: float = 1.0,
        retry_multiplier: float = 2.0,
        num_concurrent: int = 10,
        adaptive_concurrency: bool | str = False,
        max_model_length: int | None = None,
        max_frames_num: int = 10,
        chat_template_kwargs: str | dict | None = None,
        batch_size: int | str = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Unknown model_args ignored: {list(kwargs.keys())}. Check the supported parameters for the 'openai_stream' backend.")
        self.model_version = model or model_version
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        if not self.model_version or not self.base_url:
            raise ValueError("openai_stream needs the served model name and the server: pass model=<name>,base_url=<url>")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.stall_s = float(timeout) if timeout is not None else _DEFAULT_STALL_S
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.retry_multiplier = float(retry_multiplier)
        self.num_concurrent = max(1, int(num_concurrent))
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.context_budget = int(max_model_length) if max_model_length else None
        self.max_frames_num = int(max_frames_num)
        if isinstance(chat_template_kwargs, str):
            chat_template_kwargs = json.loads(chat_template_kwargs)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._progress = _ProgressClock()
        self._consecutive_stalls = 0

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("openai_stream generates only; loglikelihood is not supported")

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("openai_stream does not support multi-round generation")

    def generate_until(self, requests: list[Instance]) -> list[GenerationResult]:
        if not requests:
            return []
        return asyncio.run(self._generate(list(requests)))

    def _make_client(self) -> openai.AsyncOpenAI:
        _fit_open_files(self.num_concurrent)
        limits = httpx.Limits(max_connections=self.num_concurrent, max_keepalive_connections=self.num_concurrent, keepalive_expiry=_KEEPALIVE_EXPIRY_S)
        return openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.stall_s,
            max_retries=0,
            http_client=openai.DefaultAsyncHttpxClient(limits=limits),
        )

    async def _fetch_load(self) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                return (await http.get(_metrics_url(self.base_url))).text
        except httpx.HTTPError:
            return None

    @contextlib.asynccontextmanager
    async def _gate(self) -> AsyncIterator[asyncio.Semaphore | _AdaptiveLimiter]:
        if not self.adaptive_concurrency:
            yield asyncio.Semaphore(self.num_concurrent)
            return
        limiter = _AdaptiveLimiter(min(self.num_concurrent, _START_CONCURRENCY))
        steering = asyncio.create_task(_steer(limiter, ceiling=self.num_concurrent, fetch=self._fetch_load))
        try:
            yield limiter
        finally:
            steering.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await steering

    def _payload(self, request: Instance) -> dict:
        _, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        messages = ChatMessages(messages=doc_to_messages(self.task_dict[task][split][doc_id])).to_openai_messages(video_kwargs={"nframes": self.max_frames_num})
        max_new_tokens = gen_kwargs.get("max_new_tokens")
        payload = {"model": self.model_version, "messages": messages, "n": 1}
        if max_new_tokens:
            payload["max_tokens"] = max_new_tokens
        payload["temperature"] = gen_kwargs.get("temperature") or 0  # always sent: the server's own default is 1.0
        payload.update({name: gen_kwargs[name] for name in _STANDARD_PARAMS if gen_kwargs.get(name) is not None})

        extra_body = {sent: gen_kwargs[name] for name, sent in _EXTRA_BODY_PARAMS.items() if gen_kwargs.get(name) is not None}
        if self.context_budget is not None:
            extra_body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            text_only = all(isinstance(message["content"], str) or all(block.get("type") == "text" for block in message["content"]) for message in messages)
            room = self.context_budget - (max_new_tokens or 0)
            if text_only and max_new_tokens and room > 0:
                extra_body["truncate_prompt_tokens"] = room
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    async def _generate(self, requests: list[Instance]) -> list[GenerationResult]:
        responses: list[GenerationResult | None] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        latency = 0.0

        async def respond(client: openai.AsyncOpenAI, gate: asyncio.Semaphore | _AdaptiveLimiter, index: int) -> None:
            nonlocal latency
            async with gate:
                # Built off the event loop: media decoding would hold up every stream, and with them the stall clock.
                payload = await asyncio.to_thread(self._payload, requests[index])
                started = time.monotonic()
                responses[index] = self._result(await self._request(client, payload))
                latency += time.monotonic() - started
            pbar.update(1)

        async with self._make_client() as client, self._gate() as gate:
            tasks = [asyncio.ensure_future(respond(client, gate, index)) for index in range(len(requests))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                pbar.close()
        tokens = sum(response.token_counts.output_tokens or 0 for response in responses)
        log_metrics(total_elapsed_time=latency, total_gen_tokens=tokens, avg_speed=tokens / latency if latency else 0)
        return responses

    def _result(self, answer: _Answer | _Failed) -> GenerationResult:
        if isinstance(answer, _Failed):
            detail = answer.detail.replace("\n", " ")[:200]
            return GenerationResult(text=f"[LMMS_EVAL_REQUEST_FAILED after {answer.attempts} retries] {answer.reason}: {detail}", token_counts=TokenCounts())
        counts = TokenCounts(
            input_tokens=getattr(answer.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(answer.usage, "completion_tokens", 0) or len(answer.text.split()),
            reasoning_tokens=getattr(getattr(answer.usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0,
        )
        log_usage(model_name=self.model_version, task_name=None, source="model", **counts.to_dict())
        return GenerationResult(text=answer.text, token_counts=counts)

    async def _open_stream(self, create: Awaitable[Any]) -> Any:
        opened = time.monotonic()
        try:
            return await _await_unless_quiet(create, quiet_for=lambda: time.monotonic() - max(self._progress.last, opened), stall_s=self.stall_s)
        except _Quiet:
            raise _StreamStall(f"no chunk from any stream for {self.stall_s:g}s while opening the request", httpx.Request("POST", f"{self.base_url}/chat/completions")) from None

    async def _call(self, client: openai.AsyncOpenAI, payload: dict) -> _Answer:
        # No read timeout: `_stream_chunks` tells a queued request from a stuck one, which a per-read timer cannot.
        stream = await self._open_stream(client.chat.completions.create(**payload, stream=True, timeout=httpx.Timeout(self.stall_s, read=None)))
        return await _collect(stream, progress=self._progress, stall_s=self.stall_s, budget=self.context_budget)

    def _backoff(self, attempt: int) -> float:
        return min(_MAX_BACKOFF_S, self.retry_backoff_s * (self.retry_multiplier**attempt))

    async def _request(self, client: openai.AsyncOpenAI, payload: dict) -> _Answer | _Failed:
        connect_failures = 0
        last_error: BaseException | None = None
        # Connection retries are counted apart from backoff attempts; the bound covers whichever path is taken.
        for attempt in range(self.max_retries + _CONNECT_RETRIES + 1):
            try:
                answer = await self._call(client, payload)
                self._consecutive_stalls = 0
                return answer
            except _OverBudget as exc:
                self._consecutive_stalls = 0
                eval_logger.warning(f"{exc}. Not retrying; degrading this request.")
                return _Failed("over_budget", str(exc), attempt + 1)
            except (openai.APITimeoutError, httpx.TimeoutException) as exc:
                self._consecutive_stalls += 1
                if self._consecutive_stalls >= _STALL_ABORT_AFTER:
                    raise RuntimeError(f"vLLM server at {self.base_url} stalled on {self._consecutive_stalls} consecutive requests ({exc}); aborting the evaluation.") from exc
                eval_logger.error(f"Stall: {exc}. Not retrying; degrading this request.")
                return _Failed("stalled", _describe(exc), attempt + 1)
            except openai.APIConnectionError as exc:
                last_error = exc
                connect_failures += 1
                if connect_failures > _CONNECT_RETRIES:
                    raise RuntimeError(f"vLLM server at {self.base_url} is unreachable ({exc}); aborting the evaluation.") from exc
                eval_logger.warning(f"Connection failure: {exc}; retrying in {_CONNECT_RETRY_SLEEP_S}s")
                await asyncio.sleep(_CONNECT_RETRY_SLEEP_S)
            except openai.APIStatusError as exc:
                last_error = exc
                if exc.status_code == 429 or exc.status_code >= 500:
                    eval_logger.warning(f"HTTP {exc.status_code}; backing off {self._backoff(attempt):.1f}s")
                    await asyncio.sleep(self._backoff(attempt))
                else:
                    eval_logger.error(f"HTTP {exc.status_code}: {exc}. Not retrying; degrading this request.")
                    return _Failed("rejected", _describe(exc), attempt + 1)
            except openai.APIError as exc:
                last_error = exc
                eval_logger.warning(f"Server error in the stream: {exc}; retrying in {self._backoff(attempt):.1f}s")
                await asyncio.sleep(self._backoff(attempt))

        eval_logger.error(f"Request failed after {self.max_retries} attempts; degrading this request.")
        return _Failed("retries_exhausted", _describe(last_error), self.max_retries + _CONNECT_RETRIES + 1)
