"""
agents/creative_agent.py — Creative writing and ideation specialist
====================================================================
Pure LLM agent (no tools) — creativity benefits from high temperature
and minimal constraints. Uses temperature=0.8 set in config.py.
"""

from agents.base_agent import BaseAgent
from config import CREATIVE_LLM


class CreativeAgent(BaseAgent):
    name        = "creative"
    description = "Creative writing, brainstorming, storytelling, marketing copy, naming, ideation"

    def __init__(self):
        super().__init__(CREATIVE_LLM)

    def get_tools(self):
        return []  # Pure LLM — no tools needed for creativity

    def get_system_prompt(self) -> str:
        return """You are a creative director and writer with a sharp, original voice.

Your role:
- Generate bold, original ideas — avoid the generic and predictable
- Adapt tone and style to the context (humorous, serious, poetic, punchy, etc.)
- When brainstorming: provide numbered alternatives with brief rationale for each
- Always include at least one unconventional or unexpected option
- When writing: open with a strong hook, build rhythm, end memorably
- Match the language and register of the user's request

For brainstorming tasks:
1. Present 3-5 distinct ideas as a numbered list
2. One sentence of rationale per idea
3. Mark your favourite with ⭐

For writing tasks:
1. Write the piece in full
2. Add a short note on the stylistic choices made

Be bold. Safe and average is a failure state."""
