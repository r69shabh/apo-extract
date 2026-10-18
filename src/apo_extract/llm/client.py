"""Unified LLM client (litellm) with raw request/response logging and the
Gemini File API path for very large PDFs (bypasses litellm request-size limits).

Tests inject ``completer`` to stay fully offline; everything else is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ..config import LlmConfig
from .pdf_handler import PdfAttachment


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    raw: object = None


class LLMCallError(RuntimeError):
    def __init__(self, role: str, model: str, cause: Exception):
        super().__init__(f"LLM call failed for role={role} model={model}: {cause}")
        self.cause = cause


Completer = Callable[[dict], Awaitable[object]]


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def _response_cost(model: str, response: object, input_tokens: int, output_tokens: int) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        return _estimate_cost(model, input_tokens, output_tokens)


class LLMClient:
    """One instance per role. Logs every call to the run store with raw payloads."""

    def __init__(
        self,
        llm_config: LlmConfig,
        *,
        role: str = "extractor",
        store=None,
        run_id: Optional[str] = None,
        completer: Optional[Completer] = None,
        timeout_s: int = 180,
        tracker=None,
    ):
        self.config = llm_config
        self.role = role
        self.store = store
        self.run_id = run_id
        self.completer = completer
        self.timeout_s = timeout_s
        self.tracker = tracker
        self._call_counter = 0

    @property
    def model_string(self) -> str:
        return self.config.qualified_model

    # -- message marshalling -------------------------------------------------

    @staticmethod
    def _attach(messages: list[dict], attachment: PdfAttachment) -> list[dict]:
        import os

        if attachment.kind == "inline_pdf":
            blocks = [
                {
                    "type": "file",
                    "file": {
                        "file_data": attachment.payload,
                        "filename": os.path.basename(str(attachment.extra.get("filename", "document.pdf"))),
                    },
                }
            ]
        elif attachment.kind == "images":
            blocks = [
                {"type": "image_url", "image_url": {"url": uri}} for uri in attachment.payload
            ]
        elif attachment.kind == "text":
            blocks = [{"type": "text", "text": "Document text:\n" + str(attachment.payload)}]
        else:  # google_file handled by the direct google-genai path
            blocks = []
        out = []
        for message in messages:
            if message["role"] == "user" and blocks:
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                out.append({"role": "user", "content": list(content) + blocks})
            else:
                out.append(message)
        return out

    # -- completion ----------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        *,
        attachment: Optional[PdfAttachment] = None,
        iteration: Optional[int] = None,
        pdf_id: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        purpose: str = "extraction",
    ) -> CallResult:
        self._call_counter += 1
        started = time.time()
        call_messages = messages if attachment is None else self._attach(messages, attachment)
        request_payload = {
            "model": self.model_string,
            "messages": call_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
            "response_format": response_format,
            "purpose": purpose,
            "pdf_id": pdf_id,
            "prompt_hash": prompt_hash,
        }
        if attachment is not None and attachment.kind in ("inline_pdf", "images"):
            request_payload["attachment_kind"] = attachment.kind
            # keep raw logs small: replace data URIs with a length marker (never the live call)
            request_payload["messages"] = _shrink_data_uris(call_messages)

        try:
            if attachment is not None and attachment.kind == "google_file" and self.completer is None:
                result = await self._complete_google_file(
                    messages, attachment, temperature, max_output_tokens
                )
            elif self.completer is not None:
                response = await self.completer(request_payload)
                result = self._result_from_litellm(response, started)
            else:
                import litellm

                response = await litellm.acompletion(
                    model=self.model_string,
                    messages=call_messages,
                    temperature=request_payload["temperature"],
                    max_tokens=request_payload["max_output_tokens"],
                    response_format=response_format,
                    timeout=self.timeout_s,
                    num_retries=2,
                )
                result = self._result_from_litellm(response, started)
        except Exception as exc:
            raise LLMCallError(self.role, self.model_string, exc) from exc

        self._log(result, request_payload, iteration, pdf_id, prompt_hash)
        if self.tracker is not None:
            from .telemetry import CallRecord

            self.tracker.record(
                CallRecord(
                    role=self.role,
                    model=self.model_string,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_usd=result.cost_usd,
                    latency_ms=result.latency_ms,
                )
            )
        return result

    async def _complete_google_file(
        self,
        messages: list[dict],
        attachment: PdfAttachment,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
    ) -> CallResult:
        import os

        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY required for google_file attachments")
        started = time.time()
        client = genai.Client(api_key=api_key)
        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user_text = "\n\n".join(
            m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)
        )
        parts = [types.Part.from_uri(file_uri=attachment.payload, mime_type="application/pdf")]
        if user_text:
            parts.append(types.Part.from_text(text=user_text))
        config = types.GenerateContentConfig(
            system_instruction=system_text or None,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_output_tokens=max_output_tokens or self.config.max_output_tokens,
        )
        response = await client.aio.models.generate_content(
            model=self.config.model, contents=[types.Content(parts=parts)], config=config
        )
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        text = response.text or ""
        return CallResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model_string, input_tokens, output_tokens),
            latency_ms=int((time.time() - started) * 1000),
            raw={"usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}},
        )

    @staticmethod
    def _result_from_litellm(response: object, started: float) -> CallResult:
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        model = getattr(response, "model", "") or ""
        return CallResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_response_cost(model, response, input_tokens, output_tokens),
            latency_ms=int((time.time() - started) * 1000),
            raw=response,
        )

    def _log(self, result: CallResult, request_payload: dict, iteration, pdf_id, prompt_hash) -> None:
        if self.store is None:
            return
        stamp = f"{int(time.time() * 1000)}_{self.role}_{self._call_counter}"
        request_path = self.store.write_raw(f"req_{stamp}.json", request_payload)
        response_path = self.store.write_raw(
            f"resp_{stamp}.json",
            {
                "text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
            },
        )
        self.store.log_llm_call(
            run_id=self.run_id or "",
            role=self.role,
            model=self.model_string,
            iteration=iteration,
            prompt_hash=prompt_hash,
            pdf_id=pdf_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            raw_request_path=request_path,
            raw_response_path=response_path,
        )


def _shrink_data_uris(messages: list[dict]) -> list[dict]:
    def shrink(value):
        if isinstance(value, str) and value.startswith("data:"):
            return f"<data-uri {len(value)} bytes>"
        if isinstance(value, dict):
            return {k: shrink(v) for k, v in value.items()}
        if isinstance(value, list):
            return [shrink(v) for v in value]
        return value
    return shrink(messages)
