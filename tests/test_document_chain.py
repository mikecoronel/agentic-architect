from agentic_architect.agents.document_chain import DocumentChain, chunk_text
from agentic_architect.llm_connectors import LLMConnector

class DummyConnector(LLMConnector):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"summary{len(self.calls)}"


def test_chunk_text_merges_small_final_chunk():
    text = "word " * 4500
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 2000
    assert len(chunks[1].split()) == 2500


def test_document_chain(tmp_path):
    text = "word " * 4500
    path = tmp_path / "doc.md"
    path.write_text(text)

    connector = DummyConnector()
    chain = DocumentChain(connector)
    result = chain.analyze(str(path))

    assert len(connector.calls) == 3  # 2 chunks + final report
    assert result == "summary3"
