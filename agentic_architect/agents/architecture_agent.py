from typing import List

from ..llm_connectors import LLMConnector
from ..config import PromptConfig

class ArchitectureAgent:
    """Agent that generates a microservice architecture for banking modernization."""

    def __init__(self, llm: LLMConnector, prompts: PromptConfig):
        self.llm = llm
        self.prompts = prompts

    def generate_architecture(self, requirements: List[str]) -> str:
        prompt = self.prompts.architecture.format(
            requirements="\n".join(requirements)
        )
        return self.llm.generate(prompt)
