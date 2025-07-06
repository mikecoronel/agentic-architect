import argparse
import logging

from .config import Config
from .llm_connectors import connector_from_config
from .agents.architecture_agent import ArchitectureAgent
from .agents.review_agent import ReviewAgent


def configure_logging() -> None:
    """Configure root logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(config_path: str, req_path: str = "requerimientos.txt") -> None:
    logger = logging.getLogger(__name__)

    cfg = Config.load(config_path)
    logger.info("Configuration loaded from %s", config_path)

    with open(req_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d requirements from %s", len(requirements), req_path)

    llm = connector_from_config(cfg.llm)

    arch_agent = ArchitectureAgent(llm, cfg.prompts)
    logger.info("Generating architecture for %d requirements", len(requirements))

    architecture = arch_agent.generate_architecture(requirements)
    print("--- Proposed Architecture ---")
    print(architecture)

    if cfg.review_enabled:
        review_agent = ReviewAgent(llm, cfg.prompts)
        logger.info("Running review agent")

        review = review_agent.review(architecture)
        print("\n=== Review Report ===")
        print(review)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate banking architectures via LLM agents")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "requirements",
        nargs="?",
        default="requerimientos.txt",
        help="Path to requirements file",
    )
    args = parser.parse_args()

    configure_logging()
    run(args.config, args.requirements)

if __name__ == "__main__":
    main()
