import logging

from ..llm_connectors import LLMConnector
from ..config import PromptConfig

logger = logging.getLogger(__name__)

class ReviewAgent:
    """Agent that reviews architecture proposals."""

    def __init__(self, llm: LLMConnector, prompts: PromptConfig):
        self.llm = llm
        self.prompts = prompts

    def review(self, architecture: str) -> str:
        prompt = self.prompts.review.format(architecture=architecture)
        logger.info("Reviewing architecture")
        logger.debug("Review prompt: %s", prompt)

        return self.llm.generate(prompt)
