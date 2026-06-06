from typing import Any, Protocol, Sequence


class ChatModel(Protocol):
    def invoke(self, messages: Sequence[Any]) -> Any:
        ...

    def bind_tools(self, tools: Sequence[Any]) -> "ChatModel":
        ...


class LLMProvider(Protocol):
    def chat(
        self,
        *,
        component: str,
        model_env: str = "GEMINI_LLM_MODEL",
        fallback_model_env: str | None = None,
        default_model: str = "gemini-2.5-flash",
        timeout_env: str = "GEMINI_LLM_TIMEOUT_SECONDS",
        default_timeout: float = 20.0,
        retries_env: str = "GEMINI_LLM_RETRIES",
        default_retries: int = 1,
        tools: Sequence[Any] | None = None,
    ) -> ChatModel:
        ...
