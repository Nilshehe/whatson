import asyncio
import sys
import time
import uuid 
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import HumanMessage

import config
from orchestrator.graph import build_graph
from agents.registry import AGENT_REGISTRY
from streaming import stream_graph_synthesis


_BANNER = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          W H A T S O N  —  Multi-Agent AI                ║
║                                                          ║
║  Type your task. Whatson routes it to the right agents.  ║
║  Commands: /agents  /verbose  /help  quit                ║
╚══════════════════════════════════════════════════════════╝
"""


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_agents():
    print()
    for name, agent in AGENT_REGISTRY.items():
        tools = agent.get_tools()
        tool_str = ", ".join(t.name for t in tools) if tools else "none"
        print(f"  [{name.upper():10}] {agent.description}")
        print(f"               tools: {tool_str}")
    print()

def cmd_verbose():
    config.VERBOSE = not config.VERBOSE
    print(f"  Verbose: {'ON' if config.VERBOSE else 'OFF'}\n")

def cmd_help():
    print(__doc__)

_COMMANDS = {"/agents": cmd_agents, "/verbose": cmd_verbose, "/help": cmd_help}


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main():
    print(_BANNER)
    print("Initialising Whatson…", end="", flush=True)
    graph = build_graph()
    print(" ✓\n")
    #config
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        try:
            task = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye!")
            break
        if task.startswith("/"):
            handler = _COMMANDS.get(task.lower())
            if handler:
                handler()
            else:
                print(f"  Unknown command: {task}. Try /help\n")
            continue

        # ── Build initial state ───────────────────────────────────────────────
        snapshot = graph.get_state(thread_config)

        history = []

        if snapshot and snapshot.values:
            history = snapshot.values.get("messages", [])
        
        initial_state = {
            "messages":      history + [HumanMessage(content=task)],
            "task":          task,
            "plan":          [],
            "agent_results": {},
            "final_answer":  "",
            "approved":      False,
        }
        t0 = time.time()

        try:
            result = await stream_graph_synthesis(graph, initial_state, thread_config)
        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")
            if config.VERBOSE:
                import traceback
                traceback.print_exc()
            continue

        elapsed = time.time() - t0
        answer  = result.get("final_answer", "")

        if not answer or answer == "[Cancelled by user]":
            print()
            continue

        used = list(result.get("agent_results", {}).keys())

        # The answer has already been streamed to the terminal by stream_graph_synthesis.
        # We just print the footer.
        print("═" * 60)
        print(f"  ⏱ {elapsed:.1f}s   agents: {', '.join(used) if used else '—'}\n")


if __name__ == "__main__":
    asyncio.run(main())
