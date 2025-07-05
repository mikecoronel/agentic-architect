from typing import List

from ..llm_connectors import LLMConnector

class ArchitectureAgent:
    """Agent that generates a microservice architecture for banking modernization."""

    def __init__(self, llm: LLMConnector):
        self.llm = llm

    def generate_architecture(self, requirements: List[str]) -> str:
        prompt = (
            "You are a banking modernization expert with knowledge of microservices, "
            "cloud and on-premise platforms, container orchestrators, BIAN, TOGAF, "
            "coreless strategies and RFP best practices. Based on the following "
            "requirements, design a distributed architecture:\n" + "\n".join(requirements)
        )
        return self.llm.generate(prompt)
