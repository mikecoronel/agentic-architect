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
    """Log token usage and cost using per-million pricing."""
    total_tokens = prompt_tokens + completion_tokens
    input_price = (prompt_tokens / 1_000_000) * input_cost
    output_price = (completion_tokens / 1_000_000) * output_cost
    total_price = input_price + output_price

    logger.info("Input tokens: %d, cost: $%.6f", prompt_tokens, input_price)
    logger.info("Output tokens: %d, cost: $%.6f", completion_tokens, output_price)
    logger.info("Total tokens: %d, cost: $%.6f", total_tokens, total_price)


class LLMConnector(abc.ABC):
    """Abstract interface for connecting to various LLM providers."""

    def __init__(self, input_cost: float = 0.0, output_cost: float = 0.0) -> None:
        self.input_cost = input_cost
        self.output_cost = output_cost

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

class OpenAIConnector(LLMConnector):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", *, input_cost: float = 0.0, output_cost: float = 0.0):
        super().__init__(input_cost, output_cost)
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
        _log_usage(prompt_tokens, completion_tokens, input_cost=self.input_cost, output_cost=self.output_cost)
        return content

class AnthropicConnector(LLMConnector):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", *, input_cost: float = 0.0, output_cost: float = 0.0):
        super().__init__(input_cost, output_cost)
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
        _log_usage(prompt_tokens, completion_tokens, input_cost=self.input_cost, output_cost=self.output_cost)
        return content

class GeminiConnector(LLMConnector):
    def __init__(self, api_key: str, *, input_cost: float = 0.0, output_cost: float = 0.0):
        super().__init__(input_cost, output_cost)
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
        _log_usage(prompt_tokens, completion_tokens, input_cost=self.input_cost, output_cost=self.output_cost)
        return content

class OllamaConnector(LLMConnector):
    def __init__(self, base_url: str = "http://localhost:11434", *, input_cost: float = 0.0, output_cost: float = 0.0):
        super().__init__(input_cost, output_cost)
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
        _log_usage(prompt_tokens, completion_tokens, input_cost=self.input_cost, output_cost=self.output_cost)
        return content

def connector_from_config(cfg: Any) -> LLMConnector:
    provider = cfg.provider.lower()
    costs = getattr(cfg, "costs", {}) or {}
    input_cost = float(costs.get("input", 0.0))
    output_cost = float(costs.get("output", 0.0))
    if provider == "openai":
        logger.info("Using OpenAI connector")
        return OpenAIConnector(
            api_key=cfg.api_key,
            base_url=cfg.base_url or "https://api.openai.com/v1",
            input_cost=input_cost,
            output_cost=output_cost,
        )
    if provider == "anthropic":
        logger.info("Using Anthropic connector")
        return AnthropicConnector(
            api_key=cfg.api_key,
            base_url=cfg.base_url or "https://api.anthropic.com",
            input_cost=input_cost,
            output_cost=output_cost,
        )
    if provider == "gemini":
        logger.info("Using Gemini connector")
        return GeminiConnector(
            api_key=cfg.api_key,
            input_cost=input_cost,
            output_cost=output_cost,
        )
    if provider == "ollama":
        logger.info("Using Ollama connector")
        return OllamaConnector(
            base_url=cfg.base_url or "http://localhost:11434",
            input_cost=input_cost,
            output_cost=output_cost,
        )
    raise ValueError(f"Unknown provider {cfg.provider}")
