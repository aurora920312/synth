from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .config import Config
from .core.agent import Agent
from .llm.router import LLMRouter
from .tools import registry as tool_registry
from .memory.run_logger import RunLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentic agent")
    parser.add_argument("prompt", nargs="*", help="User prompt/goal for the agent")
    parser.add_argument("--model", default=None, help="Model id (e.g., gpt-4o-mini or local)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max reasoning steps")
    parser.add_argument(
        "--tools",
        default=None,
        help="Comma-separated list of tools to enable (default: python,fs,web)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output final result JSON with transcript path",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("Please provide a prompt, e.g.: agentic \"Search web and summarize...\"")
        sys.exit(2)

    cfg = Config.from_env()
    if args.model:
        cfg.model = args.model
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.tools is not None:
        cfg.enabled_tools = tuple(args.tools.split(","))  # type: ignore[assignment]

    runs_dir = cfg.runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    llm = LLMRouter.from_config(cfg)
    tools = tool_registry.load_tools(cfg)
    logger = RunLogger(runs_dir=runs_dir)

    agent = Agent(llm=llm, tools=tools, logger=logger, config=cfg)
    result = agent.run(prompt)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("final_answer") or "")


if __name__ == "__main__":
    main()
