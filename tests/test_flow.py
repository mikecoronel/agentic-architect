from agentic_architect.main import run
from agentic_architect.config import Config, LLMConfig, PromptConfig

class DummyConnector:
    def generate(self, prompt: str) -> str:
        return "dummy"


def test_run_invokes_review(monkeypatch, capsys):
    config = Config(
        llm=LLMConfig(provider="dummy"),
        review_enabled=True,
        prompts=PromptConfig(),
    )

    monkeypatch.setattr("agentic_architect.main.Config.load", staticmethod(lambda p: config))
    monkeypatch.setattr(
        "agentic_architect.main.connector_from_config",
        lambda cfg: DummyConnector(),
    )

    call_order = []

    def fake_generate(self, reqs):
        call_order.append("arch")
        return "ARCH"

    def fake_review(self, arch):
        call_order.append("review")
        return "REVIEW"

    monkeypatch.setattr("agentic_architect.main.ArchitectureAgent.generate_architecture", fake_generate)
    monkeypatch.setattr("agentic_architect.main.ReviewAgent.review", fake_review)

    run(["req"], "config.yaml")

    assert call_order == ["arch", "review"]
