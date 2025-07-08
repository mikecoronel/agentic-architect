import logging
from typing import List

from ..llm_connectors import LLMConnector

logger = logging.getLogger(__name__)


def chunk_text(text: str, min_tokens: int = 1000, max_tokens: int = 2000) -> List[str]:
    """Split text into chunks between min_tokens and max_tokens tokens.

    Tokens are approximated by whitespace-separated words. If the final chunk
    would be smaller than ``min_tokens`` it is merged with the previous one.
    """
    if min_tokens > max_tokens:
        raise ValueError("min_tokens cannot exceed max_tokens")

    words = text.split()
    chunks: List[List[str]] = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = words[start:end]
        if len(chunk) < min_tokens and chunks:
            chunks[-1].extend(chunk)
            break
        chunks.append(chunk)
        start = end
    return [" ".join(c) for c in chunks]


class DocumentChain:
    """Process a document in manageable chunks and combine the results."""

    def __init__(self, llm: LLMConnector, *, min_tokens: int = 1000, max_tokens: int = 2000) -> None:
        self.llm = llm
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def analyze(self, path: str) -> str:
        """Analyze a Markdown document using a simple chain-of-thought pipeline."""
        logger.info("Loading document %s", path)
        with open(path, "r") as f:
            text = f.read()

        chunks = chunk_text(text, self.min_tokens, self.max_tokens)
        logger.info("Document split into %d chunks", len(chunks))

        summaries: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            prompt = (
                "Read this section, extract the key steps, then await the next section.\n\n"
                f"{chunk}"
            )
            logger.debug("Chunk %d prompt: %s", idx, prompt)
            summaries.append(self.llm.generate(prompt))

        final_prompt = (
            "Here are all the chunk summaries—please integrate into a single coherent report.\n\n"
        )
        for i, summary in enumerate(summaries, 1):
            final_prompt += f"Chunk {i} summary:\n{summary}\n\n"

        logger.info("Generating final report from %d summaries", len(summaries))
        return self.llm.generate(final_prompt)
