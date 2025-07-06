import yaml
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMConfig:
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class PromptConfig:
    architecture: str = (
        "You are a banking modernization expert with knowledge of microservices, "
        "cloud and on-premise platforms, container orchestrators, BIAN, TOGAF, "
        "coreless strategies and RFP best practices. Based on the following "
        "requirements, design a distributed architecture:\n{requirements}"
    )
    review: str = (
        "You are a senior solutions architect. Review the following architecture "
        "for correctness, completeness and alignment with banking standards. "
        "Provide detailed feedback and recommendations.\n{architecture}"
    )

@dataclass
class Config:
    llm: LLMConfig
    review_enabled: bool = True
    prompts: PromptConfig = field(default_factory=PromptConfig)

    @staticmethod
    def load(path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        llm = LLMConfig(**data.get('llm', {}))
        prompts = PromptConfig(**data.get('prompts', {}))

        return Config(
            llm=llm,
            review_enabled=data.get('review_enabled', True),
            prompts=prompts,
        )
