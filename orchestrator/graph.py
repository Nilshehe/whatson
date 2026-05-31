"""
orchestrator/graph.py — Whatson LangGraph execution graph
==========================================================
Nodes:
  plan_node       → Router LLM breaks task into per-agent sub-tasks
  approval_node   → Human reviews/approves the plan
  dispatch_node   → Sub-agents run in PARALLEL (asyncio.gather)
  synthesize_node → Orchestrator merges all results into final answer

Streaming integration:
  dispatch_node calls stream_agent_response() for each agent so the user
  sees thinking + answer tokens live, per agent, as they run in parallel.

  synthesize_node is a plain async node — streaming is handled by
  stream_graph_synthesis() in streaming.py which calls astream_events()
  on the compiled graph and filters for node=="synthesize".
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.state import WhatsonState
from orchestrator.router import route_task
from config import ORCHESTRATOR_LLM, VERBOSE


def _registry():
    from agents.registry import AGENT_REGISTRY
    return AGENT_REGISTRY


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — PLAN
# ══════════════════════════════════════════════════════════════════════════════

async def plan_node(state: WhatsonState) -> dict:
    if VERBOSE:
        print(f"\n📋 PLANNING: '{state['task'][:80]}'")

    plan = await route_task(state["task"], _registry())
    

    if VERBOSE:
        for item in plan:
            print(f"   [{item['agent'].upper():10}] {item['subtask'][:70]}")

    return {"plan": plan, "approved": False}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — APPROVAL
# ══════════════════════════════════════════════════════════════════════════════

async def approval_node(state: WhatsonState) -> dict:
    print("\n" + "─" * 58)
    print("  WHATSON PLAN")
    print("─" * 58)
    for i, item in enumerate(state["plan"], 1):
        print(f"  {i}. [{item['agent'].upper():10}] {item['subtask']}")
    print("─" * 58)

    user_input = input("  Approve? [Enter=yes / feedback / no]: ").strip()

    if not user_input or user_input.lower() in ("yes", "y", "ja", "j", "ok"):
        print("  ✓ Approved.\n")
        return {"approved": True}

    if user_input.lower() in ("no", "n", "cancel", "nej", "avbryt"):
        print("  ✗ Cancelled.\n")
        return {"approved": False, "final_answer": "[Cancelled by user]"}

    # Feedback → re-plan
    print(f"  ↩ Re-planning with feedback: '{user_input}'\n")
    new_task = f"{state['task']}\n\n[User feedback: {user_input}]"
    new_plan = await route_task(new_task, _registry())

    print("─" * 58)
    print("  REVISED PLAN")
    print("─" * 58)
    for i, item in enumerate(new_plan, 1):
        print(f"  {i}. [{item['agent'].upper():10}] {item['subtask']}")
    print("─" * 58)

    return {"task": new_task, "plan": new_plan, "approved": True}


def _should_dispatch(state: WhatsonState) -> str:
    return "dispatch" if state.get("approved") else END


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — PARALLEL DISPATCH WITH STREAMING
# ══════════════════════════════════════════════════════════════════════════════

async def dispatch_node(state: WhatsonState) -> dict:
    """
    Run all sub-agents in parallel.
    Each agent streams its thinking + answer tokens live via stream_agent_response().

    Because agents run with asyncio.gather(), their token streams can
    interleave in the terminal — this is expected for parallel execution.
    For a cleaner display you could run them sequentially here instead.
    """
    from streaming import stream_agent_response
    registry = _registry()

    async def run_one(item: dict) -> tuple[str, str]:
        name    = item["agent"]
        subtask = item["subtask"]

        if name not in registry:
            return name, f"[ERROR] Unknown agent: '{name}'"

        agent = registry[name]

        if VERBOSE:
            print(f"  🤖 {name.upper()} starting…")

        # stream_agent_response handles thinking display + token accumulation
        result = await stream_agent_response(
            runnable=agent.get_runnable(),
            subtask=subtask,
            agent_name=name,
            history=state["messages"],
        )
        return name, result

    if VERBOSE:
        print(f"\n🚀 DISPATCH — {len(state['plan'])} agent(s) running in parallel\n")

    raw = await asyncio.gather(
        *[run_one(item) for item in state["plan"]],
        return_exceptions=True,
    )

    agent_results = {}
    for r in raw:
        if isinstance(r, Exception):
            print(f"  ❌ Agent exception: {r}")
        else:
            name, output = r
            agent_results[name] = output

    return {"agent_results": agent_results}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — SYNTHESIZE
# ══════════════════════════════════════════════════════════════════════════════

_SYNTHESIS_PROMPT = PromptTemplate.from_template("""You are Whatson. Your specialist agents have completed their work.
Merge their outputs into a single, coherent, well-structured response.

ORIGINAL TASK:
{task}

AGENT OUTPUTS:
{results}

INSTRUCTIONS:
- Combine contributions naturally — do not repeat information
- Preserve code blocks and tables exactly as produced
- Use markdown headings if the answer is multi-section
- Address the original task directly and completely
- Match the language the user wrote in
- Do not mention internal agent names or the orchestration process
""")


async def synthesize_node(state: WhatsonState) -> dict:
    """
    Merge all agent results.

    NOTE: This node's token stream is captured by stream_graph_synthesis()
    in streaming.py via astream_events(). The streaming display (thinking
    animation + answer tokens) happens there — this node just runs normally.
    """
    results = state["agent_results"]

    # Single agent — skip synthesis LLM call, return directly
    if len(results) == 1:
        answer = next(iter(results.values()))
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    if VERBOSE:
        print(f"\n🧩 SYNTHESIZING {len(results)} agent outputs…")

    results_str = "\n\n".join(
        f"[{name.upper()}]\n{output}" for name, output in results.items()
    )

    chain  = _SYNTHESIS_PROMPT | ORCHESTRATOR_LLM | StrOutputParser()
    answer = await chain.ainvoke({"task": state["task"], "results": results_str})

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(WhatsonState)

    g.add_node("plan",       plan_node)
    g.add_node("approval",   approval_node)
    g.add_node("dispatch",   dispatch_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "approval")

    g.add_conditional_edges(
        "approval",
        _should_dispatch,
        {"dispatch": "dispatch", END: END},
    )

    g.add_edge("dispatch",   "synthesize")
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=MemorySaver())
