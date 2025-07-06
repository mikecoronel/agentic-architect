# Agentic Architect

This repository provides a simple multi-agent system for generating banking
architecture proposals with the help of different large language models (LLMs).
It can connect to OpenAI, Anthropic, Google's Gemini and local models via
Ollama. The system can be configured through a YAML file and optionally uses a
second agent to review the generated architecture. Prompts for the agents
can be customized in the configuration file and the tool logs key steps
for traceability.

## Features

- **ArchitectureAgent** – generates microservices-based banking architectures
  using knowledge of cloud and on-premise platforms, container orchestrators,
  BIAN, TOGAF and coreless strategies.
- **ReviewAgent** – optionally reviews the generated architecture for accuracy
  and completeness.
- Support for multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama).
- Simple CLI for providing requirements and configuration.

## Usage

1. Install dependencies for your chosen LLM provider. For example, for OpenAI:
   ```bash
   pip install openai PyYAML
   ```
2. Create a configuration file (see `sample_config.yaml`). The file
   lets you specify API keys and customize the prompts used by the
  agents. Token usage and estimated cost are logged for each LLM request,
  providing visibility into consumption.
   Logging is configured at startup, producing messages such as:
   `2025-07-06 14:12:03 [INFO] agentic_architect.llm_connectors: Tokens used - input: 150, output: 220, total: 370, cost: $0.0200`.
3. Run the tool:
   ```bash
   python -m agentic_architect.main config.yaml "Requirement 1" "Requirement 2"
   ```

The tool prints the proposed architecture and the review if enabled.
