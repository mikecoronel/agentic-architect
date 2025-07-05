import abc
from typing import Any

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
        response = self.openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

class AnthropicConnector(LLMConnector):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic library required") from e
        self.client = anthropic.Client(api_key, base_url=base_url)

    def generate(self, prompt: str) -> str:
        response = self.client.completions.create(
            model="claude-3-opus-20240229",
            prompt=prompt,
        )
        return response.completion

class GeminiConnector(LLMConnector):
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError("google-generativeai library required") from e
        genai.configure(api_key=api_key)
        self.genai = genai

    def generate(self, prompt: str) -> str:
        model = self.genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text

class OllamaConnector(LLMConnector):
    def __init__(self, base_url: str = "http://localhost:11434"):
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError("ollama library required") from e
        self.ollama = ollama
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        response = self.ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}], base_url=self.base_url)
        return response["message"]["content"]

def connector_from_config(cfg: Any) -> LLMConnector:
    provider = cfg.provider.lower()
    if provider == "openai":
        return OpenAIConnector(api_key=cfg.api_key, base_url=cfg.base_url or "https://api.openai.com/v1")
    if provider == "anthropic":
        return AnthropicConnector(api_key=cfg.api_key, base_url=cfg.base_url or "https://api.anthropic.com")
    if provider == "gemini":
        return GeminiConnector(api_key=cfg.api_key)
    if provider == "ollama":
        return OllamaConnector(base_url=cfg.base_url or "http://localhost:11434")
    raise ValueError(f"Unknown provider {cfg.provider}")
