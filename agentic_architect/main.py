import argparse
import logging
from typing import List

from .config import Config
from .llm_connectors import connector_from_config
from .agents.architecture_agent import ArchitectureAgent
from .agents.review_agent import ReviewAgent


def run(requirements: List[str], config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = Config.load(config_path)
    logger.info("Configuration loaded from %s", config_path)
    llm = connector_from_config(cfg.llm)

    arch_agent = ArchitectureAgent(llm, cfg.prompts)
    logger.info("Generating architecture for %d requirements", len(requirements))
    cfg = Config.load(config_path)
    logger.info("Configuration loaded from %s", config_path)
    llm = connector_from_config(cfg.llm)

    arch_agent = ArchitectureAgent(llm, cfg.prompts)
    logger.info("Generating architecture for %d requirements", len(requirements))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cfg = Config.load(config_path)
    logger.info("Configuration loaded")
    llm = connector_from_config(cfg.llm)

    arch_agent = ArchitectureAgent(llm, cfg.prompts)
    architecture = arch_agent.generate_architecture(requirements)
    print("--- Proposed Architecture ---")
    print(architecture)

    if cfg.review_enabled:
        review_agent = ReviewAgent(llm, cfg.prompts)
        logger.info("Running review agent")

        review = review_agent.review(architecture)
        print("\n--- Review ---")
        print(review)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate banking architectures via LLM agents")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("requirements", nargs='+', help="List of requirement statements")
    args = parser.parse_args()

    run(args.requirements, args.config)

if __name__ == "__main__":
    main()
