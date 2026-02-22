#!/usr/bin/env python3
"""
Hybrid proxy: routes Claude Code requests between local GGUF model and OpenRouter.

- Turn < MAX_LOCAL_TURNS → local llama-server (LocoOperator-4B-GGUF)
- Turn >= MAX_LOCAL_TURNS or context exceeded → OpenRouter (Qwen3-Coder-Next)

Usage:
  # 1. Start llama-server:
  #    llama-server --model models/LocoOperator-4B-GGUF/LocoOperator-4B.gguf \
  #                 --port 8080 --ctx-size 32768 -ngl 99
  #
  # 2. Start this proxy:
  #    uv run python scripts/proxy.py
  #
  # 3. Run analysis:
  #    ANTHROPIC_BASE_URL=http://127.0.0.1:9091 claude -p "..." --dangerously-skip-permissions
"""

import json
import os
import uuid
import re
import httpx
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

LLAMA_SERVER = "http://127.0.0.1:8080"
OPENROUTER_URL = "https://openrouter.ai/api"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LISTEN_PORT = 9091

LOCAL_MODELS = {"claude-haiku-4-5-20251001"}
REMOTE_FALLBACK_MODEL = "qwen/qwen3-coder-next"
MAX_LOCAL_TURNS = 10

_local_client = httpx.Client(mounts={"http://127.0.0.1": None}, timeout=300)
_remote_client = httpx.Client(timeout=300)

TRAINING_SYSTEM_PROMPT = """You are a Claude agent, built on Anthropic's Claude Agent SDK.
You are an interactive agent that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

# Using your tools
 - To read files use Read
 - To search for files use Glob
 - To search the content of files use Grep
 - To run shell commands use Bash
 - To write files use Write
 - To edit files use Edit

For each tool call, output a JSON object within <tool_call></tool_call> tags:
<tool_call>
{"name": <tool-name>, "arguments": <args-json-object>}
</tool_call>

Tool results will be provided in <tool_response></tool_response> tags."""


def convert_anthropic_to_openai(body: dict) -> dict:
    messages = [{"role": "system", "content": TRAINING_SYSTEM_PROMPT}]
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    parts.append(f"<tool_call>\n{json.dumps({'name': block['name'], 'arguments': block.get('input', {})})}\n</tool_call>")
                elif block.get("type") == "tool_result":
                    rc = block.get("content", "")
                    if isinstance(rc, list):
                        rc = " ".join(b.get("text", "") for b in rc)
                    parts.append(f"<tool_response>\n{rc}\n</tool_response>")
            messages.append({"role": role, "content": "\n".join(parts)})
    return {
        "model": "loco-operator-4b",
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 0.0),
        "stream": False,
    }


def parse_tool_calls(text: str) -> tuple[str, list]:
    tool_uses = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    def replace(m):
        try:
            obj = json.loads(m.group(1))
            tool_uses.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": obj.get("name", ""),
                "input": obj.get("arguments", obj.get("input", {})),
            })
        except Exception:
            pass
        return ""
    remaining = pattern.sub(replace, text).strip()
    return remaining, tool_uses


def build_anthropic_content(text: str) -> tuple[list, str]:
    remaining, tool_uses = parse_tool_calls(text)
    content = []
    if remaining:
        content.append({"type": "text", "text": remaining})
    content.extend(tool_uses)
    stop_reason = "tool_use" if tool_uses else "end_turn"
    return content, stop_reason


def make_anthropic_sse(text: str, msg_id: str) -> str:
    content, stop_reason = build_anthropic_content(text)
    out = []
    out.append(f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'loco-operator-4b', 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n")
    for i, block in enumerate(content):
        out.append(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': block})}\n\n")
        if block["type"] == "text":
            out.append(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'text_delta', 'text': block['text']}})}\n\n")
        elif block["type"] == "tool_use":
            out.append(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}})}\n\n")
        out.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n")
    out.append(f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n")
    out.append(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n")
    return "".join(out)


def openai_to_anthropic_response(oai: dict) -> dict:
    text = oai["choices"][0]["message"]["content"] or ""
    usage = oai.get("usage", {})
    content, stop_reason = build_anthropic_content(text)
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": "loco-operator-4b",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[proxy] {format % args}")

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw)
        model = body.get("model", "")
        is_stream = body.get("stream", False)
        msgs = body.get("messages", [])
        turn = len([m for m in msgs if m["role"] == "assistant"])
        use_local = model in LOCAL_MODELS and turn < MAX_LOCAL_TURNS
        print(f"[proxy] model={model} turn={turn} local={use_local} msgs={len(msgs)}")

        if use_local:
            self._handle_local(body, is_stream)
        else:
            if model in LOCAL_MODELS and turn >= MAX_LOCAL_TURNS:
                print(f"[proxy] turn limit ({turn}), fallback to OpenRouter")
                body = dict(body)
                body["model"] = REMOTE_FALLBACK_MODEL
                raw = json.dumps(body).encode()
            self._handle_remote(raw, is_stream)

    def _handle_remote(self, raw: bytes, is_stream: bool):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_KEY}",
            }
            for k, v in self.headers.items():
                if k.lower() in ("anthropic-version", "anthropic-beta", "x-api-key"):
                    headers[k] = v
            resp = _remote_client.post(f"{OPENROUTER_URL}/v1/messages", content=raw, headers=headers)
            print(f"[proxy] remote status={resp.status_code}")
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() in ("content-type", "cache-control"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception as e:
            print(f"[proxy] remote error: {e}")
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_local(self, body: dict, is_stream: bool):
        oai_body = convert_anthropic_to_openai(body)
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        try:
            resp = _local_client.post(
                f"{LLAMA_SERVER}/v1/chat/completions",
                json=oai_body,
                headers={"Content-Type": "application/json"},
            )
            oai_json = resp.json()
            if "choices" not in oai_json:
                err = oai_json.get("error", {})
                print(f"[proxy] llama-server error: {oai_json}")
                if err.get("type") == "exceed_context_size_error":
                    print("[proxy] context exceeded, fallback to OpenRouter")
                    fallback = dict(body)
                    fallback["model"] = REMOTE_FALLBACK_MODEL
                    self._handle_remote(json.dumps(fallback).encode(), is_stream)
                    return
                raise KeyError("choices")
            text = oai_json["choices"][0]["message"]["content"] or ""
            print(f"[proxy] local output: {text[:200]}")
            if is_stream:
                sse = make_anthropic_sse(text, msg_id)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(sse.encode())
            else:
                out = json.dumps(openai_to_anthropic_response(oai_json)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        except Exception as e:
            print(f"[proxy] local error: {e}")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


if __name__ == "__main__":
    if not OPENROUTER_KEY:
        print("WARNING: OPENROUTER_API_KEY not set in .env")
    server = HTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
    print(f"Proxy listening on http://127.0.0.1:{LISTEN_PORT}")
    print(f"Local: llama-server at {LLAMA_SERVER} (turns < {MAX_LOCAL_TURNS})")
    print(f"Remote: OpenRouter {REMOTE_FALLBACK_MODEL}")
    server.serve_forever()
