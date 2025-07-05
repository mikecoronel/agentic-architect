import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@dataclass
class Config:
    llm: LLMConfig
    review_enabled: bool = True

    @staticmethod
    def load(path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        llm = LLMConfig(**data.get('llm', {}))
        return Config(llm=llm, review_enabled=data.get('review_enabled', True))
