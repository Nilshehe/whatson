"""
orchestrator/state.py — Shared LangGraph state
===============================================
WhatsonState is passed between every node in the graph.
Each node receives the full state and returns a dict with only
the keys it modified — LangGraph merges updates automatically.

Annotated[List[BaseMessage], operator.add]:
  When two nodes write to "messages" concurrently (e.g. during
  parallel dispatch), their lists are concatenated rather than
  one overwriting the other. operator.add = list concatenation.
"""

import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage


class WhatsonState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], operator.add]
    task:     str           # Original user question, unchanged throughout

    # ── Routing ──────────────────────────────────────────────────────────────
    plan: list
    # Router output: [{"agent": "research", "subtask": "..."}, ...]

    # ── Execution ─────────────────────────────────────────────────────────────
    agent_results: dict
    # Keyed by agent name: {"research": "...", "code": "..."}

    # ── Output ───────────────────────────────────────────────────────────────
    final_answer: str       # Synthesized answer shown to the user

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    approved: bool          # Set True by approval_node to proceed to dispatch
