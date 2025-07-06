import logging
from agentic_architect.llm_connectors import LLMConnector, _log_usage
from agentic_architect.agents.architecture_agent import ArchitectureAgent
from agentic_architect.config import PromptConfig

class DummyConnector(LLMConnector):
    def generate(self, prompt: str) -> str:
        _log_usage(5, 7, input_cost=0.01, output_cost=0.02)
        return "dummy"

def test_usage_logging(caplog):
    caplog.set_level(logging.INFO)
    connector = DummyConnector()
    prompts = PromptConfig()
    agent = ArchitectureAgent(connector, prompts)
    agent.generate_architecture(["requirement"])
    assert any("Input tokens" in rec.message for rec in caplog.records)
    assert any("Output tokens" in rec.message for rec in caplog.records)
    assert any("Total tokens" in rec.message for rec in caplog.records)
