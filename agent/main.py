import argparse
import os
import sys

from .agent import ProgrammingExpertAgent
from .llm_client import LLMClient
from .tools import default_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Programming Expert Agent (HTTP LLM)")
    parser.add_argument("task", type=str, nargs="?", help="Task prompt for the agent")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "Qwen3-coder-480B-A35B-Instruct-FP8"))
    parser.add_argument("--endpoint", type=str, default=os.getenv("INFERENCE_URL", "http://localhost:8000/api/inference"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "20")))
    parser.add_argument("--quiet", action="store_true", help="Reduce step output")

    args = parser.parse_args()

    if not args.task:
        print("Please provide a task. Example:")
        print("python -m agent.main \"Refactor the utils parser to avoid quadratic behavior.\"")
        return 2

    client = LLMClient(endpoint_url=args.endpoint, model=args.model, temperature=args.temperature)
    tools = default_registry()
    agent = ProgrammingExpertAgent(client=client, tools=tools, max_steps=args.max_steps)

    final = agent.run(task=args.task, verbose=(not args.quiet))
    print("\n=== FINAL ANSWER ===\n")
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
