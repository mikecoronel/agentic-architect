import abc
import logging
from typing import Any

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - library optional
    tiktoken = None

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    """Return an approximate token count for a text."""
    if tiktoken is not None:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    return len(text.split())


def _log_usage(prompt_tokens: int, completion_tokens: int, input_cost: float, output_cost: float) -> None:
    total_tokens = prompt_tokens + completion_tokens
    cost = (prompt_tokens / 1000) * input_cost + (completion_tokens / 1000) * output_cost
    logger.info(
        "Tokens used - input: %d, output: %d, total: %d, cost: $%.4f",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost,
    )


class LLMConnector(abc.ABC):
    """Abstract interface for connecting to various LLM providers."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

class OpenAIConnector(LLMConnector):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("openai library required") from e
        self.openai = openai
        self.openai.api_key = api_key
        self.openai.base_url = base_url

    def generate(self, prompt: str) -> str:
        logger.info("OpenAI request")
        logger.debug("Prompt: %s", prompt)

        response = self.openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None) or {}
        prompt_tokens = usage.get("prompt_tokens") or _count_tokens(prompt)
        completion_tokens = usage.get("completion_tokens") or _count_tokens(content)
        _log_usage(prompt_tokens, completion_tokens, input_cost=0.03, output_cost=0.06)
        return content

class AnthropicConnector(LLMConnector):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic library required") from e
        self.client = anthropic.Client(api_key, base_url=base_url)

    def generate(self, prompt: str) -> str:
        logger.info("Anthropic request")
        logger.debug("Prompt: %s", prompt)

        response = self.client.completions.create(
            model="claude-3-opus-20240229",
            prompt=prompt,
        )
        content = response.completion
        usage = getattr(response, "usage", None) or {}
        prompt_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or _count_tokens(prompt)
        )
        completion_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or _count_tokens(content)
        )
        _log_usage(prompt_tokens, completion_tokens, input_cost=0.015, output_cost=0.075)
        return content

class GeminiConnector(LLMConnector):
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError("google-generativeai library required") from e
        genai.configure(api_key=api_key)
        self.genai = genai

    def generate(self, prompt: str) -> str:
        logger.info("Gemini request")
        logger.debug("Prompt: %s", prompt)

        model = self.genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        content = response.text
        prompt_tokens = _count_tokens(prompt)
        completion_tokens = _count_tokens(content)
        _log_usage(prompt_tokens, completion_tokens, input_cost=0.0, output_cost=0.0)
        return content

class OllamaConnector(LLMConnector):
    def __init__(self, base_url: str = "http://localhost:11434"):
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError("ollama library required") from e
        self.ollama = ollama
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        logger.info("Ollama request")
        logger.debug("Prompt: %s", prompt)

        response = self.ollama.chat(
            model="llama2",
            messages=[{"role": "user", "content": prompt}],
            base_url=self.base_url,
        )
        content = response["message"]["content"]
        prompt_tokens = _count_tokens(prompt)
        completion_tokens = _count_tokens(content)
        _log_usage(prompt_tokens, completion_tokens, input_cost=0.0, output_cost=0.0)
        return content

def connector_from_config(cfg: Any) -> LLMConnector:
    provider = cfg.provider.lower()
    if provider == "openai":
        logger.info("Using OpenAI connector")
        return OpenAIConnector(api_key=cfg.api_key, base_url=cfg.base_url or "https://api.openai.com/v1")
    if provider == "anthropic":
        logger.info("Using Anthropic connector")
        return AnthropicConnector(api_key=cfg.api_key, base_url=cfg.base_url or "https://api.anthropic.com")
    if provider == "gemini":
        logger.info("Using Gemini connector")
        return GeminiConnector(api_key=cfg.api_key)
    if provider == "ollama":
        logger.info("Using Ollama connector")
        return OllamaConnector(base_url=cfg.base_url or "http://localhost:11434")
    raise ValueError(f"Unknown provider {cfg.provider}")
