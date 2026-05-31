"""
agents/registry.py — Central agent registry
============================================
AGENT_REGISTRY is the single source of truth for all available agents.

The router reads agent names and descriptions from this dict to decide
which agents to dispatch for a given user task.

To add a new agent:
  1. Create agents/my_agent.py inheriting BaseAgent
  2. Import and instantiate it here
  3. Add it to AGENT_REGISTRY — the router picks it up automatically
"""

from agents.research_agent import ResearchAgent
from agents.code_agent     import CodeAgent
from agents.data_agent     import DataAgent
from agents.creative_agent import CreativeAgent
from agents.general_agent  import GeneralAgent
from agents.printer_agent import BambuAgent

AGENT_REGISTRY: dict = {
    "research": ResearchAgent(),
    "code":     CodeAgent(),
    "data":     DataAgent(),
    "creative": CreativeAgent(),
    "general":  GeneralAgent(),
    "printer": BambuAgent(),
}
