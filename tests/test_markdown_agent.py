from agentic_architect.agents.markdown_agent import MarkdownAnalysisAgent
from agentic_architect.llm_connectors import LLMConnector


class DummyConnector(LLMConnector):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"analysis{len(self.calls)}"


def test_analyze_large_markdown(tmp_path):
    text = "word " * 6000
    path = tmp_path / "doc.md"
    path.write_text(text)

    connector = DummyConnector()
    agent = MarkdownAnalysisAgent(connector)
    result = agent.analyze("Summarize", path=str(path))

    assert len(connector.calls) == 4  # 3 chunks + final summary
    assert result == "analysis4"


def test_analyze_prompt_only():
    connector = DummyConnector()
    agent = MarkdownAnalysisAgent(connector)

    result = agent.analyze("Hello world")

    assert connector.calls == ["Hello world"]
    assert result == "analysis1"
