"""Stable user-facing handling for LLM provider outages."""

LLM_API_ERROR_MESSAGE = "❌ LLM API 暫時無法使用，請稍後再試。"


class LLMAPIUnavailableError(RuntimeError):
    pass


def is_llm_api_error(error: BaseException | str) -> bool:
    if isinstance(error, str):
        return "LLMAPIUnavailableError" in error
    if isinstance(error, LLMAPIUnavailableError):
        return True
    return type(error).__module__.startswith(("google.genai", "openai"))
