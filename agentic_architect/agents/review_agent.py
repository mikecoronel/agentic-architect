from ..llm_connectors import LLMConnector
from ..config import PromptConfig

class ReviewAgent:
    """Agent that reviews architecture proposals."""

    def __init__(self, llm: LLMConnector, prompts: PromptConfig):
        self.llm = llm
        self.prompts = prompts

    def review(self, architecture: str) -> str:
        prompt = self.prompts.review.format(architecture=architecture)
        return self.llm.generate(prompt)
