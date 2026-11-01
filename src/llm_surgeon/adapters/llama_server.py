"""One llama-server per GGUF export: the model loads once for all of stage 09's prompts.

On a B300 stage 09 started 32 llama-completion processes for two 2B GGUF exports; each took ~2.3 s, of
which generating 32 tokens took ~40 ms (docs/implementation-log.md). A 27B Q8_0 export reads ~27 GB per load.

Interface verified at b11889 (d854d98), tools/server/README.md: ``GET /health`` answers 200 once the model
is loaded; ``POST /completion`` takes ``prompt`` as a string (special tokens are parsed and BOS is added
only when the model's ``tokenizer.ggml.add_bos_token`` is true, as in llama-completion), and returns
``content``, ``stop_type`` (eos | limit | word), ``tokens_evaluated`` and ``tokens_predicted``.
``cache_prompt`` is off because reused KV entries are not guaranteed bit-identical to a fresh evaluation.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..errors import Interrupted
from ..subprocesses import BackgroundProcess, CommandRunner

FINISH = {"eos": "stop", "word": "stop", "limit": "length"}
BIND_ERROR = "couldn't bind HTTP server socket"


class LlamaServerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_argv(cmd: list[str], model: Path, ctx: int, gpu_layers: str, port: int, threads: int) -> list[str]:
    """One slot (``-np 1``): requests run one at a time, each in a fresh, unbatched context like llama-completion."""
    return [
        *cmd,
        "-m",
        str(model),
        "-c",
        str(ctx),
        "-ngl",
        gpu_layers,
        "-t",
        str(max(1, threads)),
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
    ]


def completion_request(prompt: str, n_predict: int) -> dict[str, Any]:
    """Greedy decoding with the same settings as ``llama-completion --temp 0 -s 0``."""
    return {"prompt": prompt, "n_predict": int(n_predict), "temperature": 0, "seed": 0, "cache_prompt": False}


def completion_result(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "response": body.get("content", ""),
        "finish_reason": FINISH.get(body.get("stop_type"), body.get("stop_type")),
        "prompt_tokens": body.get("tokens_evaluated"),
        "new_tokens": body.get("tokens_predicted"),
        "truncated": bool(body.get("truncated")),
    }


class LlamaServer:
    """Context manager: start, wait for /health, serve completions, stop. Failures raise LlamaServerError."""

    def __init__(
        self,
        runner: CommandRunner,
        cmd: list[str],
        model: Path,
        *,
        ctx: int,
        gpu_layers: str,
        threads: int,
        env: dict[str, str] | None,
        log_dir: Path,
        name: str,
        timeout_seconds: float,
        interrupt_event: threading.Event,
    ) -> None:
        self.runner, self.cmd, self.model = runner, cmd, model
        self.ctx, self.gpu_layers, self.threads = ctx, gpu_layers, threads
        self.env, self.log_dir, self.name = env, log_dir, name
        self.timeout_seconds = timeout_seconds
        self.interrupt_event = interrupt_event
        self.port = 0
        self.process: BackgroundProcess | None = None
        self.load_seconds: float | None = None
        self.result = None
        # Never route loopback requests through a proxy from the environment.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _failure(self, message: str) -> LlamaServerError:
        tail = " | ".join(self.process.tail[-6:]) if self.process else ""
        return LlamaServerError(f"{message}; log {self.log_dir / (self.name + '.stderr.log')}: {tail}")

    def _start(self) -> None:
        self.port = free_port()
        self.process = self.runner.start(
            server_argv(self.cmd, self.model, self.ctx, self.gpu_layers, self.port, self.threads),
            cwd=None,
            env=self.env,
            log_dir=self.log_dir,
            name=self.name,
        )

    def __enter__(self) -> LlamaServer:
        started = time.monotonic()
        self._start()
        rebinds = 0
        try:
            while True:
                if self.interrupt_event.is_set():
                    raise Interrupted()
                if not self.process.alive():
                    # The free port can be taken between free_port() and the server's bind (server-http.cpp
                    # logs "couldn't bind HTTP server socket"); that happens before the model loads, so retry.
                    self.process.stop()
                    if rebinds < 2 and any(BIND_ERROR in line for line in self.process.tail):
                        rebinds += 1
                        self._start()
                        continue
                    raise self._failure(f"llama-server exited with {self.process.proc.returncode} before the model loaded")
                if time.monotonic() - started > self.timeout_seconds:
                    raise self._failure(f"llama-server did not load the model within {self.timeout_seconds:.0f} s")
                try:
                    with self._opener.open(self._url("/health"), timeout=5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    pass  # not listening yet, or 503 while the model loads
                time.sleep(0.2)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.load_seconds = round(time.monotonic() - started, 1)
        return self

    def complete(self, prompt: str, n_predict: int) -> dict[str, Any]:
        if self.interrupt_event.is_set():
            raise Interrupted()
        data = json.dumps(completion_request(prompt, n_predict)).encode("utf-8")
        request = urllib.request.Request(self._url("/completion"), data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise self._failure(f"/completion failed: {exc}") from exc
        return completion_result(body)

    def __exit__(self, *exc: object) -> None:
        if self.process is not None and self.result is None:
            self.result = self.process.stop()
