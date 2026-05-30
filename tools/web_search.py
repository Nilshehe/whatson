"""
tools/web_search.py — DuckDuckGo web search tool
=================================================
Uses the `ddgs` package (successor to duckduckgo-search).
Install: pip install ddgs

The tool is a plain LangChain @tool that sub-agents bind to their LLM.
Results are returned as a formatted string with title + snippet per hit.
"""

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return the top results.

    Args:
        query:       Natural language search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted string with result titles, URLs and snippets.
        Returns an error message string if the search fails.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return "web_search unavailable: run `pip install ddgs` to enable."

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        return f"web_search error: {exc}"

    if not results:
        return f"No results found for: {query}"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href  = r.get("href",  "")
        body  = r.get("body",  "").strip()
        lines.append(f"[{i}] {title}\n    URL: {href}\n    {body}\n")

    return "\n".join(lines)
