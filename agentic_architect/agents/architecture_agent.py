from typing import List

import logging

from ..llm_connectors import LLMConnector
from ..config import PromptConfig

logger = logging.getLogger(__name__)

class ArchitectureAgent:
    """Agent that generates a microservice architecture for banking modernization."""

    def __init__(self, llm: LLMConnector, prompts: PromptConfig):
        self.llm = llm
        self.prompts = prompts

    def generate_architecture(self, requirements: List[str]) -> str:
        prompt = self.prompts.architecture.format(
            requirements="\n".join(requirements)
        )
        logger.info("Generating architecture")
        logger.debug("Architecture prompt: %s", prompt)
        return self.llm.generate(prompt)
