"""
agents/research_agent.py — Research and information specialist
==============================================================
Uses DuckDuckGo web search to find current, real-world information.
Runs a full ReAct loop: the LLM decides what to search and when to stop.
"""

from agents.base_agent import BaseAgent
from tools.web_search import web_search
from config import RESEARCH_LLM


class ResearchAgent(BaseAgent):
    name        = "research"
    description = "Facts, current events, news, explanations, background information, anything that benefits from web search"

    def __init__(self):
        super().__init__(RESEARCH_LLM)

    def get_tools(self):
        return [web_search]

    def get_system_prompt(self) -> str:
        return """You are a research specialist with access to DuckDuckGo web search.

Your role:
- Find accurate, up-to-date information using web_search when needed
- Synthesize multiple sources into a clear, structured answer
- Use markdown headings (##) for multi-section responses
- Clearly distinguish between confirmed facts and uncertain claims
- Include relevant examples and analogies to aid understanding
- Respond in the same language the user used

Search strategy:
- Start with a broad query, then narrow if needed
- Run 2-3 searches for complex topics to cross-validate
- Prefer primary sources (official docs, papers) over aggregators

Avoid:
- Speculating without marking it as such
- Repeating information already found
- Vague non-answers"""
