"""Fake llama.cpp tools for CPU tests: fake_llama.py <convert|quantize|imatrix|server> ARGS.

FAKE_QUANT_FAIL=<TYPE> makes llama-quantize fail for that type; FAKE_MMPROJ_FAIL=1 breaks --mmproj;
FAKE_SERVER_FAIL=1 makes llama-server exit before the model loads; FAKE_SERVER_MMPROJ_FAIL=1 makes it exit when
started with --mmproj; FAKE_SERVER_BIND_FAIL_ONCE=<marker path>
makes its first start fail to bind the port.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fakeutil import count, degraded, fake_answer, log, minimal_gguf  # noqa: E402


def arg(flag: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1]


def serve(model: str, port: int, threads: str, vision: bool) -> int:
    """llama-server at d854d98 as stage 09 uses it (tools/server/README.md): GET /health, POST /completion
    with a string prompt (content, stop_type, tokens_evaluated, tokens_predicted) and, with --mmproj,
    POST /v1/chat/completions with an image (OpenAI-style choices[0].message.content)."""
    if os.environ.get("FAKE_SERVER_FAIL"):
        print("main: error: failed to load model (fake failure)", file=sys.stderr)
        return 1
    if vision and os.environ.get("FAKE_SERVER_MMPROJ_FAIL"):
        print("clip_init: failed to load the multimodal projector (fake failure)", file=sys.stderr)
        return 1
    marker = os.environ.get("FAKE_SERVER_BIND_FAIL_ONCE")  # a path: the first start loses its port race
    if marker and not Path(marker).exists():
        Path(marker).write_text("1")
        print(f"srv    start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: {port}", file=sys.stderr)
        return 1
    log("llama-server", f"start model={Path(model).name} threads={threads} vision={vision}")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep stderr quiet like the real server at default verbosity
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"status": "ok"}) if self.path == "/health" else self.reply(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/chat/completions" and vision:
                image = body["messages"][0]["content"][0]["image_url"]["url"]
                assert image.startswith("data:image/png;base64,") and body["chat_template_kwargs"] == {"enable_thinking": False}
                count("llama-server-image")
                return self.reply(200, {"choices": [{"message": {"role": "assistant", "content": "Red."}, "finish_reason": "stop"}]})
            if self.path != "/completion":
                return self.reply(404, {})
            assert body["temperature"] == 0 and body["cache_prompt"] is False, "greedy, no prompt cache"
            prompt = body["prompt"]
            count("llama-server-request")
            if "<|im_start|>" in prompt:  # a rendered comparison prompt
                content = fake_answer(prompt, degraded(model))
            else:  # a raw loader smoke prompt
                content = " The answer is Paris, and 17 * 23 = 391."
            self.reply(
                200,
                {"content": content, "stop_type": "eos", "tokens_evaluated": len(prompt.split()), "tokens_predicted": 9, "truncated": False},
            )

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"main: server is listening on http://127.0.0.1:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    tool = sys.argv[1]
    rest = sys.argv[2:]
    if "--version" in rest:
        print("version: 11889 (d854d98) fake")
        return 0
    if "--help" in rest:
        print(f"usage: llama-{tool} [options] (fake)")
        return 1
    if "--print-supported-models" in rest:
        print("Qwen3_5ForConditionalGeneration")
        return 0
    count(f"llama-{tool}")
    if tool == "convert":
        out = Path(arg("--outfile"))
        if "--mmproj" in rest:
            if os.environ.get("FAKE_MMPROJ_FAIL"):
                print("NotImplementedError: vision tower not supported", file=sys.stderr)
                return 1
            minimal_gguf(out, {"general.architecture": "clip", "general.type": "mmproj"})  # real converter keeps --outfile
        else:
            minimal_gguf(
                out,
                {
                    "general.architecture": "qwen35",
                    "tokenizer.ggml.model": "gpt2",
                    "tokenizer.chat_template": "{{ messages }}",
                    "general.file_type": arg("--outtype"),
                },
            )
        return 0
    if tool == "quantize":
        args = [a for a in rest if a != "--imatrix"]
        if "--imatrix" in rest:
            args.remove(arg("--imatrix"))
        src, dst, qtype = Path(args[0]), Path(args[1]), args[2]
        if os.environ.get("FAKE_QUANT_FAIL", "").upper() == qtype.upper():
            print(f"llama_model_quantize: failed to quantize: {qtype} (fake failure)", file=sys.stderr)
            return 1
        minimal_gguf(
            dst,
            {
                "general.architecture": "qwen35",
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.chat_template": "{{ messages }}",
                "general.file_type": qtype,
                "fake.source_size": str(src.stat().st_size),
            },
        )
        return 0
    if tool == "imatrix":
        shutil.copyfile(arg("-f"), arg("-o"))
        return 0
    if tool == "server":
        return serve(arg("-m"), int(arg("--port")), arg("-t"), "--mmproj" in rest)
    print(f"unknown fake tool {tool}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
