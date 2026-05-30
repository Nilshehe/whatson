"""
config.py — Whatson configuration
==================================
reasoning=True  → qwen3 streams <think> tokens separately in
                  chunk.additional_kwargs["reasoning_content"]
                  The main chunk.content is the clean answer only.

reasoning=None  → (default) model decides; <think>...</think> tags
                  appear inline in chunk.content — harder to parse.

Set reasoning=True for the cleanest streaming split.
"""

from langchain_ollama import ChatOllama

ORCHESTRATOR_LLM = ChatOllama(model="qwen3:4b", temperature=0,   reasoning=True)
RESEARCH_LLM     = ChatOllama(model="qwen3:4b", temperature=0.2, reasoning=True)
CODE_LLM         = ChatOllama(model="qwen3:4b", temperature=0.1, reasoning=True)
DATA_LLM         = ChatOllama(model="qwen3:4b", temperature=0.0, reasoning=True)
CREATIVE_LLM     = ChatOllama(model="qwen3:4b", temperature=0.8, reasoning=True)
GENERAL_LLM      = ChatOllama(model="qwen3:4b", temperature=0.4, reasoning=True)

VERBOSE = True
