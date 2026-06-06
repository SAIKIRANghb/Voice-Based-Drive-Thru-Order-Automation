import json
import logging
import os
import re
from typing import Any, Sequence

from voice_agent.config import get_gemini_api_key
from voice_agent.llm.interfaces import ChatModel, LLMProvider


DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_LLM_TIMEOUT_SECONDS = 20.0


class GeminiLLMProvider:
    def chat(
        self,
        *,
        component: str,
        model_env: str = "GEMINI_LLM_MODEL",
        fallback_model_env: str | None = None,
        default_model: str = DEFAULT_GEMINI_LLM_MODEL,
        timeout_env: str = "GEMINI_LLM_TIMEOUT_SECONDS",
        default_timeout: float = DEFAULT_GEMINI_LLM_TIMEOUT_SECONDS,
        retries_env: str = "GEMINI_LLM_RETRIES",
        default_retries: int = 1,
        tools: Sequence[Any] | None = None,
    ) -> ChatModel:
        model_name = os.getenv(model_env) or (os.getenv(fallback_model_env) if fallback_model_env else None) or default_model
        request_timeout = float(os.getenv(timeout_env, str(default_timeout)))
        request_timeout = max(request_timeout, 10.0)
        retries = int(os.getenv(retries_env, str(default_retries)))

        from langchain_google_genai import ChatGoogleGenerativeAI

        logging.info("Calling Gemini LLM model '%s' for %s.", model_name, component)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=get_gemini_api_key(),
            request_timeout=request_timeout,
            retries=retries,
        )
        return llm.bind_tools(tools) if tools else llm


_llm_provider: LLMProvider = GeminiLLMProvider()


def get_llm_provider() -> LLMProvider:
    return _llm_provider


def set_llm_provider(provider: LLMProvider) -> None:
    global _llm_provider
    _llm_provider = provider


def response_text(response: Any) -> str:
    content = response.content
    if isinstance(content, list):
        text = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content).strip()
    else:
        text = str(content or "").strip()
    return text


def response_json(response: Any) -> dict:
    text = response_text(response)
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)
