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

    from io import StringIO

    def fake_open(path, *args, **kwargs):
        if path == "requerimiento.txt":
            return StringIO("req1\nreq2\n")
        return open_orig(path, *args, **kwargs)

    open_orig = open
    monkeypatch.setattr("builtins.open", fake_open)

    run("config.yaml", "requerimiento.txt")

    assert call_order == ["arch", "review"]
