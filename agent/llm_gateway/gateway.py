"""
LLM Gateway V7 — Multi-provider router with failover + embedding endpoint.

Provider priority: Bedrock Sonnet (reliable) → NVIDIA (fallback)

Auto-routes by task:
- perception → Bedrock Claude Sonnet 4.6 (structured JSON)
- decision   → Bedrock Claude Sonnet 4.6 (tool-calling)
- memory     → Bedrock Claude Sonnet 4.6 (classification)
- fallback   → NVIDIA llama-3.3-70b

Embedding:
- primary    → Ollama nomic-embed-text (768-d, local)
- fallback   → Gemini gemini-embedding-001 (768-d via outputDimensionality)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from logger import get_logger

log = get_logger("gateway")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

EMBED_OLLAMA_URL = os.getenv("EMBED_OLLAMA_URL", "http://localhost:11434")
EMBED_OLLAMA_MODEL = os.getenv("EMBED_OLLAMA_MODEL", "nomic-embed-text")
EMBED_FALLBACK_MODEL = os.getenv("EMBED_FALLBACK_MODEL", "gemini-embedding-001")
EMBED_DIMENSION = 768


@dataclass
class GatewayResponse:
    text: str | None = None
    tool_calls: list[dict] | None = None
    parsed: dict | None = None
    provider: str = ""
    model: str = ""
    tier: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    is_error: bool = False
    error_transient: bool = False


@dataclass
class GatewayClient:
    nvidia_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    aws_profile: str = field(default_factory=lambda: settings.aws_profile)
    aws_region: str = field(default_factory=lambda: settings.aws_region)

    def __post_init__(self):
        # NVIDIA (fallback, disabled)
        self.nvidia_client = None
        if self.nvidia_key:
            import openai
            self.nvidia_client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=self.nvidia_key)

        # Bedrock: use 'bedrock' profile credentials via aws login
        self.bedrock = None
        self._cred_expiry = 0
        self._init_bedrock()

    def _init_bedrock(self):
        """Initialize Bedrock client using credentials from `aws login --profile bedrock`.

        Uses `aws configure export-credentials` to get temporary credentials,
        since the 'bedrock' profile uses login_session (aws login) which
        boto3's native profile loading doesn't support directly.
        """
        import boto3
        import subprocess
        from datetime import datetime

        try:
            result = subprocess.run(
                ["aws", "configure", "export-credentials", "--profile", self.aws_profile, "--format", "process"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                cred_data = json.loads(result.stdout)
                session = boto3.Session(
                    aws_access_key_id=cred_data["AccessKeyId"],
                    aws_secret_access_key=cred_data["SecretAccessKey"],
                    aws_session_token=cred_data.get("SessionToken"),
                    region_name=self.aws_region,
                )
                self.bedrock = session.client("bedrock-runtime")

                # Track credential expiry for auto-refresh
                expiry_str = cred_data.get("Expiration", "")
                if expiry_str:
                    try:
                        self._cred_expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        self._cred_expiry = time.time() + 3600
                else:
                    self._cred_expiry = time.time() + 3600

                log.info("bedrock_init", method="export-credentials", profile=self.aws_profile)
                return
        except Exception:
            pass

        # Fallback: default credential chain (env vars, instance role, etc.)
        try:
            self.bedrock = boto3.client("bedrock-runtime", region_name=self.aws_region)
            log.info("bedrock_init", method="default-chain")
        except Exception as e:
            log.error("bedrock_init_failed", error=str(e)[:100])

    def _refresh_bedrock(self):
        """Refresh Bedrock client with fresh credentials from the bedrock profile."""
        try:
            import boto3
            import subprocess
            from datetime import datetime

            result = subprocess.run(
                ["aws", "configure", "export-credentials", "--profile", self.aws_profile, "--format", "process"],
                capture_output=True, text=True, timeout=5,
            )

            if result.returncode != 0:
                # Credentials expired — trigger re-login
                subprocess.run(
                    ["aws", "login", "--profile", self.aws_profile, "--no-browser"],
                    capture_output=True, text=True, timeout=30,
                )
                result = subprocess.run(
                    ["aws", "configure", "export-credentials", "--profile", self.aws_profile, "--format", "process"],
                    capture_output=True, text=True, timeout=5,
                )

            if result.returncode == 0:
                cred_data = json.loads(result.stdout)
                session = boto3.Session(
                    aws_access_key_id=cred_data["AccessKeyId"],
                    aws_secret_access_key=cred_data["SecretAccessKey"],
                    aws_session_token=cred_data.get("SessionToken"),
                    region_name=self.aws_region,
                )
                self.bedrock = session.client("bedrock-runtime")

                expiry_str = cred_data.get("Expiration", "")
                if expiry_str:
                    try:
                        self._cred_expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00")).timestamp()
                    except:
                        self._cred_expiry = time.time() + 600
                else:
                    self._cred_expiry = time.time() + 600
                log.info("bedrock_refresh", profile=self.aws_profile)
        except Exception as e:
            log.error("bedrock_refresh_failed", error=str(e)[:100])

    def _start_refresh_timer(self):
        """Background thread that refreshes credentials every 10 minutes."""
        import threading

        def _refresh_loop():
            while True:
                import time as _time
                _time.sleep(600)  # refresh every 10 minutes
                self._refresh_bedrock()

        t = threading.Thread(target=_refresh_loop, daemon=True)
        t.start()

    def _ensure_credentials(self):
        """Refresh credentials if expired or about to expire."""
        if self._cred_expiry and time.time() > self._cred_expiry - 60:
            self._refresh_bedrock()

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
        auto_route: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> GatewayResponse:
        start = time.time()
        self._ensure_credentials()

        # Bedrock only
        if self.bedrock:
            resp = self._call_bedrock(messages, tools, tool_choice, response_format, temperature, max_tokens)
            resp.latency_ms = (time.time() - start) * 1000
            if not resp.is_error:
                self._trace(auto_route, messages, tools, resp)
                return resp
            # If error looks like expired credentials, refresh and retry once
            if "expired" in (resp.text or "").lower() or "security token" in (resp.text or "").lower():
                self._refresh_bedrock()
                resp = self._call_bedrock(messages, tools, tool_choice, response_format, temperature, max_tokens)
                resp.latency_ms = (time.time() - start) * 1000
                if not resp.is_error:
                    self._trace(auto_route, messages, tools, resp)
            return resp

        # All fallbacks disabled — Bedrock only
        if False and self.nvidia_client:
            resp = self._call_nvidia(messages, tools, tool_choice, response_format, temperature)
            if not resp.is_error:
                resp.latency_ms = (time.time() - start) * 1000
                self._trace(auto_route, messages, tools, resp)
                return resp
            log.info("nvidia_failed_trying_gemini", error=resp.text[:80] if resp.text else "")

        # Gemini fallback disabled — Bedrock only
        if False:
            pass

        return GatewayResponse(is_error=True, text="[gateway error: no providers configured]")

    def _call_nvidia(
        self, messages, tools, tool_choice, response_format, temperature
    ) -> GatewayResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
        }

        if tools and not response_format:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        if response_format and "schema" in response_format:
            kwargs["response_format"] = {"type": "json_object"}
            schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(response_format['schema'], indent=2)}\n\nRespond with ONLY the JSON, no other text."
            msgs = [m.copy() for m in messages]
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += schema_hint
            else:
                msgs.append({"role": "user", "content": schema_hint})
            kwargs["messages"] = msgs

        try:
            response = self.nvidia_client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            return GatewayResponse(model=NVIDIA_MODEL, is_error=True, error_transient=True,
                                   text=f"[gateway error: NVIDIA: {e}]")
        except Exception as e:
            return GatewayResponse(model=NVIDIA_MODEL, is_error=True,
                                   text=f"[gateway error: NVIDIA: {e}]")

        resp = GatewayResponse(model=NVIDIA_MODEL, provider="nvidia")
        choice = response.choices[0]

        if choice.message.tool_calls:
            resp.tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                resp.tool_calls.append({"name": tc.function.name, "arguments": args})
        elif choice.message.content:
            resp.text = choice.message.content
            if response_format:
                try:
                    resp.parsed = json.loads(resp.text)
                except json.JSONDecodeError:
                    text = resp.text
                    start_idx = text.find("{")
                    end_idx = text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        try:
                            resp.parsed = json.loads(text[start_idx:end_idx])
                        except json.JSONDecodeError:
                            pass

        if response.usage:
            resp.input_tokens = response.usage.prompt_tokens
            resp.output_tokens = response.usage.completion_tokens

        return resp

    def vision(
        self,
        messages: list[dict],
        *,
        image_bytes: bytes,
        image_format: str = "png",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> GatewayResponse:
        """Send image + text to Bedrock Claude Sonnet for vision tasks."""
        self._ensure_credentials()
        start = time.time()

        text_content = ""
        for msg in messages:
            if msg.get("role") in ("user", "system"):
                text_content += msg.get("content", "") + "\n"

        bedrock_messages = [{
            "role": "user",
            "content": [
                {"text": text_content.strip()},
                {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
            ],
        }]

        kwargs: dict[str, Any] = {
            "modelId": BEDROCK_MODEL,
            "messages": bedrock_messages,
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens},
        }

        try:
            response = self.bedrock.converse(**kwargs)
        except Exception as e:
            log.error("bedrock_vision_error", model=BEDROCK_MODEL, error=str(e))
            return GatewayResponse(model=BEDROCK_MODEL, is_error=True,
                                   error_transient="throttl" in str(e).lower(),
                                   text=f"[gateway error: Bedrock vision: {e}]")

        resp = self._parse_bedrock_response(response, None)
        resp.latency_ms = (time.time() - start) * 1000
        return resp

    def _call_bedrock(
        self, messages, tools, tool_choice, response_format, temperature, max_tokens=1024
    ) -> GatewayResponse:
        bedrock_messages, system_text = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": BEDROCK_MODEL,
            "messages": bedrock_messages,
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens},
        }

        if system_text:
            kwargs["system"] = [{"text": system_text}]

        if tools and not response_format:
            kwargs["toolConfig"] = {
                "tools": [self._convert_tool(t) for t in tools],
            }
            if tool_choice == "auto":
                kwargs["toolConfig"]["toolChoice"] = {"auto": {}}

        if response_format and "schema" in response_format:
            schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(response_format['schema'], indent=2)}\n\nRespond with ONLY the JSON, no other text."
            if system_text:
                kwargs["system"] = [{"text": system_text + schema_hint}]
            else:
                kwargs["system"] = [{"text": schema_hint}]

        try:
            response = self.bedrock.converse(**kwargs)
        except Exception as e:
            log.error("bedrock_error", model=BEDROCK_MODEL, error=str(e))
            return GatewayResponse(model=BEDROCK_MODEL, is_error=True,
                                   error_transient="throttl" in str(e).lower(),
                                   text=f"[gateway error: Bedrock: {e}]")

        return self._parse_bedrock_response(response, response_format)

    def _parse_bedrock_response(self, response: dict, response_format: dict | None) -> GatewayResponse:
        resp = GatewayResponse(model=BEDROCK_MODEL, provider="bedrock")

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        tool_calls = []
        text_parts = []

        for block in content_blocks:
            if "toolUse" in block:
                tc = block["toolUse"]
                tool_calls.append({"name": tc["name"], "arguments": tc.get("input", {})})
            elif "text" in block:
                text_parts.append(block["text"])

        if tool_calls:
            resp.tool_calls = tool_calls
        if text_parts:
            resp.text = "\n".join(text_parts)

        if response_format and resp.text:
            try:
                resp.parsed = json.loads(resp.text)
            except json.JSONDecodeError:
                text = resp.text
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        resp.parsed = json.loads(text[start_idx:end_idx])
                    except json.JSONDecodeError:
                        pass

        usage = response.get("usage", {})
        resp.input_tokens = usage.get("inputTokens", 0)
        resp.output_tokens = usage.get("outputTokens", 0)

        return resp

    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict], str]:
        bedrock_msgs = []
        system_text = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text += content + "\n"
                continue
            bedrock_role = "user" if role == "user" else "assistant"
            bedrock_msgs.append({"role": bedrock_role, "content": [{"text": content}]})

        if bedrock_msgs:
            merged = [bedrock_msgs[0]]
            for msg in bedrock_msgs[1:]:
                if msg["role"] == merged[-1]["role"]:
                    merged[-1]["content"].extend(msg["content"])
                else:
                    merged.append(msg)
            bedrock_msgs = merged

        return bedrock_msgs, system_text.strip()

    def _convert_tool(self, tool: dict) -> dict:
        params = tool.get("parameters", {})
        params = {k: v for k, v in params.items() if k != "additionalProperties"}
        return {
            "toolSpec": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {"json": params if params else {"type": "object", "properties": {}}},
            }
        }

    def chat_stream(self, messages: list[dict], *, temperature: float = 0.3):
        """Stream response tokens from Bedrock. Yields text chunks."""
        bedrock_messages, system_text = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": BEDROCK_MODEL,
            "messages": bedrock_messages,
            "inferenceConfig": {"temperature": temperature, "maxTokens": 1024},
        }
        if system_text:
            kwargs["system"] = [{"text": system_text}]
        try:
            response = self.bedrock.converse_stream(**kwargs)
            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
        except Exception as e:
            yield f"[error: {e}]"

    def _call_gemini_chat(
        self, messages, tools, tool_choice, response_format, temperature
    ) -> GatewayResponse:
        try:
            from google import genai
            from google.genai import types
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            client = genai.Client(api_key=gemini_key)
            model_id = "gemini-2.0-flash"

            # Build contents from messages
            system_text = ""
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_text += content + "\n"
                else:
                    gemini_role = "user" if role == "user" else "model"
                    contents.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)]))

            # Add schema hint for structured output
            if response_format and "schema" in response_format:
                schema_hint = f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(response_format['schema'], indent=2)}\n\nRespond with ONLY the JSON, no other text."
                if contents and contents[-1].role == "user":
                    last_text = contents[-1].parts[0].text + schema_hint
                    contents[-1] = types.Content(role="user", parts=[types.Part.from_text(text=last_text)])

            # Build tool declarations for Gemini
            gemini_tools = None
            if tools and not response_format:
                func_declarations = []
                for t in tools:
                    params = t.get("parameters", {})
                    clean_params = {k: v for k, v in params.items() if k != "additionalProperties"}
                    func_declarations.append(types.Tool(function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"],
                            description=t.get("description", ""),
                            parameters=clean_params if clean_params else None,
                        )
                    ]))
                gemini_tools = func_declarations

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024,
                system_instruction=system_text if system_text else None,
                tools=gemini_tools,
            )

            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )

            resp = GatewayResponse(model=model_id, provider="gemini")

            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        args = dict(fc.args) if fc.args else {}
                        resp.tool_calls = resp.tool_calls or []
                        resp.tool_calls.append({"name": fc.name, "arguments": args})
                    elif hasattr(part, 'text') and part.text:
                        resp.text = (resp.text or "") + part.text

            if response_format and resp.text:
                try:
                    resp.parsed = json.loads(resp.text)
                except json.JSONDecodeError:
                    text = resp.text
                    start_idx = text.find("{")
                    end_idx = text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        try:
                            resp.parsed = json.loads(text[start_idx:end_idx])
                        except json.JSONDecodeError:
                            pass

            return resp
        except Exception as e:
            log.error("gemini_chat_error", error=str(e)[:100])
            return GatewayResponse(model="gemini-2.0-flash", is_error=True,
                                   text=f"[gateway error: Gemini: {e}]")

    def _trace(self, auto_route, messages, tools, resp):
        try:
            from tracer import trace_llm_call
            trace_llm_call(
                role=auto_route or "default", model=resp.model, messages=messages,
                tools=tools, response_text=resp.text, tool_calls=resp.tool_calls,
                is_error=resp.is_error, latency_ms=resp.latency_ms,
                tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
            )
        except Exception:
            pass

    def embed(self, text: str, *, task_type: str = "retrieval_document") -> list[float] | None:
        """Get a 768-d embedding vector. Gemini primary (better quality), Ollama fallback."""
        vec = self._embed_gemini(text, task_type=task_type)
        if vec is not None:
            return vec
        vec = self._embed_ollama(text)
        return vec

    def _embed_ollama(self, text: str) -> list[float] | None:
        import httpx
        try:
            resp = httpx.post(
                f"{EMBED_OLLAMA_URL}/api/embed",
                json={"model": EMBED_OLLAMA_MODEL, "input": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                vec = embeddings[0]
                if len(vec) >= EMBED_DIMENSION:
                    return vec[:EMBED_DIMENSION]
                return vec
        except Exception as e:
            log.debug("ollama_embed_failed", error=str(e)[:80])
        return None

    def _embed_gemini(self, text: str, *, task_type: str = "retrieval_document") -> list[float] | None:
        try:
            from google import genai
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if not gemini_key:
                return None
            client = genai.Client(api_key=gemini_key)
            result = client.models.embed_content(
                model=EMBED_FALLBACK_MODEL,
                contents=text,
                config={
                    "task_type": task_type,
                    "output_dimensionality": EMBED_DIMENSION,
                },
            )
            if result and result.embeddings:
                return list(result.embeddings[0].values)
        except Exception as e:
            log.debug("gemini_embed_failed", error=str(e)[:80])
        return None


gateway = GatewayClient()
