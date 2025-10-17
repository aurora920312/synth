from __future__ import annotations

import json
import os
import subprocess
import sys


def run_agent(args: list[str]) -> str:
    env = os.environ.copy()
    env["AGENTIC_MODEL"] = "local"
    res = subprocess.run([sys.executable, "-m", "agentic", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def test_echo_finalizes():
    out = run_agent(["--json", "Say hi to me."])
    data = json.loads(out)
    assert data["final_answer"].lower().startswith("say hi")


def test_arithmetic_uses_python_tool():
    out = run_agent(["--json", "Compute 2 + 3 * 4"])
    data = json.loads(out)
    assert data["final_answer"] != ""
