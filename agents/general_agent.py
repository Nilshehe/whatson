"""
agents/general_agent.py — General-purpose fallback agent
=========================================================
Handles tasks that don't clearly fit other specialists.
Has access to all tools as a safety net.
"""

from agents.base_agent import BaseAgent
from tools.web_search    import web_search
from tools.calculator    import calculator
from tools.code_executor import execute_python
from config import GENERAL_LLM


class GeneralAgent(BaseAgent):
    name        = "general"
    description = "Mixed tasks, summarization, translation, Q&A, anything that doesn't fit other specialists"

    def __init__(self):
        super().__init__(GENERAL_LLM)

    def get_tools(self):
        return [web_search, calculator, execute_python]

    def get_system_prompt(self) -> str:
        return """You are Whatson — a helpful, knowledgeable and reliable AI assistant.

Your role:
- Answer questions accurately and clearly
- Use web_search when current or factual information is needed
- Use calculator for any numerical computations
- Use execute_python for data processing or scripting tasks
- Adapt depth and tone to the complexity of the question
- Respond in the same language the user used

Be honest about uncertainty. Prefer useful, concrete answers over hedged non-answers."""
