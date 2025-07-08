import logging
from typing import List

from ..llm_connectors import LLMConnector

logger = logging.getLogger(__name__)


def _chunk_text(text: str, max_tokens: int = 2048) -> List[str]:
    """Split text into chunks based on approximate token count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i : i + max_tokens])
        chunks.append(chunk)
    return chunks


class MarkdownAnalysisAgent:
    """Agent that analyzes large markdown files using chain-of-thought."""

    def __init__(self, llm: LLMConnector):
        self.llm = llm

    def analyze(
        self, user_prompt: str, path: str | None = None, *, max_tokens: int = 2048
    ) -> str:
        """Analyze a markdown file with the provided user prompt."""
        if not path:
            logger.info("No file provided, sending prompt directly")
            return self.llm.generate(user_prompt)

        logger.info("Loading markdown file from %s", path)
        with open(path, "r") as f:
            text = f.read()

        chunks = _chunk_text(text, max_tokens=max_tokens)
        logger.info("File split into %d chunks", len(chunks))

        thoughts = []
        for idx, chunk in enumerate(chunks, 1):
            prompt = (
                f"{user_prompt}\n\n"
                f"### Chunk {idx}/{len(chunks)}\n"
                f"{chunk}\n\n"
                "Provide a short chain-of-thought analysis of this chunk."
            )
            logger.debug("Chunk %d prompt: %s", idx, prompt)
            thoughts.append(self.llm.generate(prompt))

        summary_prompt = "Combine the following analyses into a final answer:\n\n"
        for i, thought in enumerate(thoughts, 1):
            summary_prompt += f"Chunk {i} analysis:\n{thought}\n\n"
        summary_prompt += "Provide the overall analysis."

        logger.info("Generating final summary")
        return self.llm.generate(summary_prompt)
