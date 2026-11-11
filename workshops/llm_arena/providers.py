"""LLM provider adapters (HTTP APIs, subscription CLIs, and a mock).

Adapted from the preliminary arena (jsafrata/calhunger-llm,
``arena/providers.py``).  Changes for the benchmark harness:

- Decoupled from the arena's TOML config: providers take an
  ``LLMSeatConfig`` (per-seat identity) and ``ProviderRuntime``
  (timeouts/retries), both plain dataclasses built by the CLI.
- ``ProviderReply`` now carries token usage where the API reports it
  (input_tokens/output_tokens), so per-decision cost is part of the
  trajectory record — the workshop's trajectory-level assessment axis.
- The Codex CLI provider materializes the action schema from
  ``action_protocol.ACTION_SCHEMA`` instead of a checked-in file, so the
  schema can never drift from the parser.
- The mock provider clamps to the env's real ``MAX_QTY``.

API keys are read from environment variables only and are never logged.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._paths import ARENA_ROOT, bootstrap

bootstrap()

from env.actions import MAX_QTY  # noqa: E402

from .action_protocol import ACTION_SCHEMA, ANTHROPIC_ACTION_SCHEMA  # noqa: E402


SUPPORTED_PROVIDERS = (
    "openai_api",
    "anthropic_api",
    "gemini_api",
    "deepseek_api",
    "codex_cli",
    "claude_cli",
    "mock",
)

DEFAULT_KEY_ENVS = {
    "openai_api": "OPENAI_API_KEY",
    "anthropic_api": "ANTHROPIC_API_KEY",
    "gemini_api": "GEMINI_API_KEY",
    "deepseek_api": "DEEPSEEK_API_KEY",
}


@dataclass(frozen=True)
class LLMSeatConfig:
    """Identity of one LLM seat: provider, model, and optional overrides."""

    label: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    executable: str | None = None


@dataclass(frozen=True)
class ProviderRuntime:
    decision_timeout_seconds: float = 120.0
    http_attempts: int = 3
    project_root: Path = ARENA_ROOT


class ProviderError(RuntimeError):
    """A provider could not produce a response."""


@dataclass(frozen=True)
class ProviderReply:
    text: str
    provider: str
    model: str
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Prompt-cache hits reported by the provider (subset of the input;
    # billed at the provider's cached rate).
    cached_input_tokens: int | None = None


class DecisionProvider:
    def __init__(self, seat: LLMSeatConfig, runtime: ProviderRuntime) -> None:
        self.seat = seat
        self.runtime = runtime

    def preflight(self) -> list[str]:
        return []

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        """Produce one reply.  ``state`` is the structured game state the
        prompt was rendered from — only the mock provider consumes it;
        real providers must decide from the prompt text alone."""
        raise NotImplementedError


def _required_key(seat: LLMSeatConfig) -> tuple[str, str | None]:
    name = seat.api_key_env or DEFAULT_KEY_ENVS.get(seat.provider, "")
    return name, os.environ.get(name) if name else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _http_post_once(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_headers = {k.lower(): v for k, v in response.headers.items()}
            return json.loads(response.read().decode("utf-8")), response_headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = body[-2000:] if body else str(exc)
        error = ProviderError(f"HTTP {exc.code}: {message}")
        error.status = exc.code  # type: ignore[attr-defined]
        error.retry_after = exc.headers.get("Retry-After")  # type: ignore[attr-defined]
        raise error from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"network error: {exc.reason}") from None
    except TimeoutError:
        raise ProviderError("HTTP request timed out") from None
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned non-JSON HTTP data: {exc.msg}") from None


class HTTPProvider(DecisionProvider):
    async def _post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        last_error: ProviderError | None = None
        for attempt in range(self.runtime.http_attempts):
            try:
                return await asyncio.to_thread(
                    _http_post_once,
                    url=url,
                    headers=headers,
                    payload=payload,
                    timeout=self.runtime.decision_timeout_seconds,
                )
            except ProviderError as exc:
                last_error = exc
                status = getattr(exc, "status", None)
                retryable = status is None or status == 429 or 500 <= status < 600
                # Anthropic's structured-output grammar compiler can time
                # out under concurrent load; it reports HTTP 400 but is
                # transient — retry it like a 5xx.
                if status == 400 and "Grammar compilation timed out" in str(exc):
                    retryable = True
                if not retryable or attempt + 1 >= self.runtime.http_attempts:
                    raise
                raw_retry = getattr(exc, "retry_after", None)
                try:
                    delay = float(raw_retry) if raw_retry is not None else 2**attempt
                except ValueError:
                    delay = 2**attempt
                await asyncio.sleep(min(30.0, max(0.1, delay)))
        assert last_error is not None
        raise last_error


class OpenAIAPIProvider(HTTPProvider):
    def preflight(self) -> list[str]:
        name, value = _required_key(self.seat)
        return [] if value else [f"missing environment variable {name}"]

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        name, key = _required_key(self.seat)
        if not key:
            raise ProviderError(f"missing environment variable {name}")
        base = (self.seat.base_url or "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": self.seat.model,
            "input": prompt,
            # Game actions are tiny; minimal reasoning keeps the output
            # budget for the JSON decision (the API analogue of the
            # Codex CLI's model_reasoning_effort="none").  Without it, a
            # reasoning model intermittently spends the whole budget on
            # reasoning items and returns no output text.
            "reasoning": {"effort": "none"},
            "max_output_tokens": 512,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "calhunger_action",
                    "strict": True,
                    "schema": ACTION_SCHEMA,
                }
            },
        }
        data, headers = await self._post(
            url=f"{base}/responses",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        text = data.get("output_text")
        if not isinstance(text, str):
            pieces: list[str] = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    value = content.get("text")
                    if isinstance(value, str):
                        pieces.append(value)
            text = "".join(pieces)
        if not text:
            raise ProviderError("OpenAI response contained no output text")
        usage = data.get("usage", {}) or {}
        details = usage.get("input_tokens_details", {}) or {}
        return ProviderReply(
            text=text,
            provider=self.seat.provider,
            model=self.seat.model,
            request_id=headers.get("x-request-id") or data.get("id"),
            input_tokens=_int_or_none(usage.get("input_tokens")),
            output_tokens=_int_or_none(usage.get("output_tokens")),
            cached_input_tokens=_int_or_none(details.get("cached_tokens")),
        )


class AnthropicAPIProvider(HTTPProvider):
    def preflight(self) -> list[str]:
        name, value = _required_key(self.seat)
        return [] if value else [f"missing environment variable {name}"]

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        name, key = _required_key(self.seat)
        if not key:
            raise ProviderError(f"missing environment variable {name}")
        base = (self.seat.base_url or "https://api.anthropic.com").rstrip("/")
        payload = {
            "model": self.seat.model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": ANTHROPIC_ACTION_SCHEMA,
                }
            },
        }
        data, headers = await self._post(
            url=f"{base}/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        pieces = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        text = "".join(pieces)
        if not text:
            raise ProviderError("Anthropic response contained no output text")
        usage = data.get("usage", {}) or {}
        # input_tokens excludes cache reads/writes; fold writes into the
        # input count (billed at 1.25x) and report reads separately.
        base_in = _int_or_none(usage.get("input_tokens")) or 0
        created = _int_or_none(usage.get("cache_creation_input_tokens")) or 0
        cached = _int_or_none(usage.get("cache_read_input_tokens"))
        return ProviderReply(
            text=text,
            provider=self.seat.provider,
            model=self.seat.model,
            request_id=headers.get("request-id") or data.get("id"),
            input_tokens=base_in + created,
            output_tokens=_int_or_none(usage.get("output_tokens")),
            cached_input_tokens=cached,
        )


class GeminiAPIProvider(HTTPProvider):
    def preflight(self) -> list[str]:
        name, value = _required_key(self.seat)
        return [] if value else [f"missing environment variable {name}"]

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        name, key = _required_key(self.seat)
        if not key:
            raise ProviderError(f"missing environment variable {name}")
        base = (
            self.seat.base_url
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        model = urllib.parse.quote(self.seat.model, safe="")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
                "responseJsonSchema": ACTION_SCHEMA,
            },
        }
        # Gemini intermittently returns an empty candidate under load;
        # that's a transient generation glitch, not a protocol failure —
        # retry in place instead of aborting the game.
        text = ""
        for empty_retry in range(self.runtime.http_attempts):
            data, headers = await self._post(
                url=f"{base}/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                payload=payload,
            )
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                if text:
                    break
            await asyncio.sleep(min(10.0, 2.0 * (empty_retry + 1)))
        else:
            feedback = data.get("promptFeedback", {})
            raise ProviderError(
                f"Gemini returned no output text after retries: {feedback}"
            )
        usage = data.get("usageMetadata", {}) or {}
        return ProviderReply(
            text=text,
            provider=self.seat.provider,
            model=self.seat.model,
            request_id=headers.get("x-request-id"),
            input_tokens=_int_or_none(usage.get("promptTokenCount")),
            output_tokens=_int_or_none(usage.get("candidatesTokenCount")),
            cached_input_tokens=_int_or_none(
                usage.get("cachedContentTokenCount")
            ),
        )


class DeepSeekAPIProvider(HTTPProvider):
    def preflight(self) -> list[str]:
        name, value = _required_key(self.seat)
        return [] if value else [f"missing environment variable {name}"]

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        name, key = _required_key(self.seat)
        if not key:
            raise ProviderError(f"missing environment variable {name}")
        base = (self.seat.base_url or "https://api.deepseek.com").rstrip("/")
        payload = {
            "model": self.seat.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            # V4 enables thinking by default.  Game actions are
            # deliberately tiny; disabling it avoids spending the output
            # budget before the JSON decision is emitted.
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        data, headers = await self._post(
            url=f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("DeepSeek response contained no choice text") from None
        if not isinstance(text, str) or not text:
            reasoning = message.get("reasoning_content") or ""
            raise ProviderError(
                "DeepSeek response contained empty output text "
                f"(finish_reason={choice.get('finish_reason')!r}, "
                f"reasoning_chars={len(reasoning)})"
            )
        usage = data.get("usage", {}) or {}
        return ProviderReply(
            text=text,
            provider=self.seat.provider,
            model=self.seat.model,
            request_id=headers.get("x-request-id") or data.get("id"),
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            cached_input_tokens=_int_or_none(
                usage.get("prompt_cache_hit_tokens")
            ),
        )


async def _run_process(
    command: list[str], *, prompt: str, timeout: float, cwd: Path
) -> tuple[str, str]:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ProviderError(f"could not start {command[0]}: {exc}") from None
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(prompt.encode("utf-8")), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ProviderError(f"{command[0]} timed out after {timeout:g}s") from None
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")
    if process.returncode != 0:
        detail = (err or out).strip()[-2000:]
        raise ProviderError(f"{command[0]} exited {process.returncode}: {detail}")
    return out, err


def _find_executable(configured: str | None, default: str) -> str | None:
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return str(configured_path)
        return shutil.which(configured)
    found = shutil.which(default)
    if found:
        return found
    if default == "codex":
        mac_app = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
        if mac_app.is_file():
            return str(mac_app)
    return None


class CodexCLIProvider(DecisionProvider):
    def _executable(self) -> str | None:
        return _find_executable(self.seat.executable, "codex")

    def preflight(self) -> list[str]:
        return [] if self._executable() else ["codex executable was not found"]

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        executable = self._executable()
        if not executable:
            raise ProviderError("codex executable was not found")
        with tempfile.TemporaryDirectory(prefix="calhunger-codex-") as temp_dir:
            schema = Path(temp_dir) / "action.schema.json"
            schema.write_text(json.dumps(ACTION_SCHEMA, indent=2))
            output = Path(temp_dir) / "last-message.json"
            command = [
                executable,
                "exec",
                "--model",
                self.seat.model,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--config",
                'model_reasoning_effort="none"',
                "--output-schema",
                str(schema),
                "--output-last-message",
                str(output),
                "-C",
                str(self.runtime.project_root),
                "-",
            ]
            stdout, _ = await _run_process(
                command,
                prompt=prompt,
                timeout=self.runtime.decision_timeout_seconds,
                cwd=self.runtime.project_root,
            )
            text = output.read_text(encoding="utf-8") if output.exists() else stdout
        if not text.strip():
            raise ProviderError("Codex CLI returned an empty final message")
        return ProviderReply(text, self.seat.provider, self.seat.model)


class ClaudeCLIProvider(DecisionProvider):
    def _executable(self) -> str | None:
        return _find_executable(self.seat.executable, "claude")

    def preflight(self) -> list[str]:
        return [] if self._executable() else [
            "claude executable was not found; install Claude Code and sign in"
        ]

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        executable = self._executable()
        if not executable:
            raise ProviderError("claude executable was not found")
        command = [
            executable,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.seat.model,
            "--max-turns",
            "1",
        ]
        stdout, _ = await _run_process(
            command,
            prompt=prompt,
            timeout=self.runtime.decision_timeout_seconds,
            cwd=self.runtime.project_root,
        )
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            text = stdout
        else:
            text = envelope.get("result", stdout) if isinstance(envelope, dict) else stdout
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Claude CLI returned an empty final message")
        return ProviderReply(text, self.seat.provider, self.seat.model)


class MockProvider(DecisionProvider):
    """Deterministic market maker used to test the orchestration offline."""

    async def decide(
        self, prompt: str, state: dict[str, Any] | None = None
    ) -> ProviderReply:
        del prompt  # the mock decides from the structured state
        if state is None:
            raise ProviderError("mock provider requires the structured state")

        decision = {
            "action": "wait",
            "side": None,
            "food": None,
            "quantity": None,
            "price": None,
        }
        decision.update(self._choose(state))
        return ProviderReply(
            json.dumps(decision, separators=(",", ":")),
            self.seat.provider,
            self.seat.model,
        )

    @staticmethod
    def _choose(state: dict[str, Any]) -> dict[str, Any]:
        produces = list(state["produces"])
        consumes = list(state["consumes"])
        inventory = state["food_units"]
        own_quotes = state["own_open_quotes"]
        other_quotes = state["quotes_from_other_players"]

        reserved_cash = sum(
            int(quote["price"]) * int(quote["quantity"])
            for quote in own_quotes
            if quote["side"] == "bid"
        )
        available_cash = int(state["cash"]) - reserved_cash
        reserved_food = {food: 0 for food in inventory}
        for quote in own_quotes:
            if quote["side"] == "ask":
                reserved_food[quote["food"]] += int(quote["quantity"])
        available = {
            food: int(quantity) - reserved_food[food]
            for food, quantity in inventory.items()
        }

        required = sorted(consumes, key=lambda food: (available[food], food))
        for food in required:
            if int(available[food]) >= 30:
                continue
            asks = [
                quote
                for quote in other_quotes
                if quote["side"] == "ask" and quote["food"] == food
            ]
            if asks:
                best_price = min(int(quote["price"]) for quote in asks)
                affordable = available_cash // max(1, best_price)
                quantity = min(MAX_QTY, affordable)
                if quantity > 0:
                    return {
                        "action": "take",
                        "side": "bid",
                        "food": food,
                        "quantity": quantity,
                    }

        own_asks = {
            quote["food"] for quote in own_quotes if quote["side"] == "ask"
        }
        for food in produces:
            surplus = int(available[food]) - 10
            if food not in own_asks and surplus > 0:
                return {
                    "action": "post",
                    "side": "ask",
                    "food": food,
                    "quantity": min(MAX_QTY, surplus),
                    "price": 1,
                }
        return {"action": "wait"}


def make_provider(seat: LLMSeatConfig, runtime: ProviderRuntime) -> DecisionProvider:
    providers: dict[str, type[DecisionProvider]] = {
        "openai_api": OpenAIAPIProvider,
        "anthropic_api": AnthropicAPIProvider,
        "gemini_api": GeminiAPIProvider,
        "deepseek_api": DeepSeekAPIProvider,
        "codex_cli": CodexCLIProvider,
        "claude_cli": ClaudeCLIProvider,
        "mock": MockProvider,
    }
    if seat.provider not in providers:
        raise ValueError(
            f"unsupported provider {seat.provider!r}; "
            f"choose one of {sorted(providers)}"
        )
    return providers[seat.provider](seat, runtime)


def load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE file without replacing exported variables."""
    if not path.exists():
        return
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"{path}:{line_number}: invalid environment name")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
