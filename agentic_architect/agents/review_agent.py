from ..llm_connectors import LLMConnector

class ReviewAgent:
    """Agent that reviews architecture proposals."""

    def __init__(self, llm: LLMConnector):
        self.llm = llm

    def review(self, architecture: str) -> str:
        prompt = (
            "You are a senior solutions architect. Review the following architecture "
            "for correctness, completeness and alignment with banking standards. "
            "Provide detailed feedback and recommendations.\n" + architecture
        )
        return self.llm.generate(prompt)
