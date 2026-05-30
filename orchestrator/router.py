import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config import ORCHESTRATOR_LLM
import sys

_ROUTER_PROMPT = PromptTemplate.from_template("""You are Whatson — an intelligent AI orchestrator that delegates tasks to specialist agents.

AVAILABLE AGENTS:
{agents_desc}

USER TASK:
{task}

INSTRUCTIONS:
1. Analyse the task carefully
2. Select ONLY the agents genuinely needed (not necessarily all of them)
3. Split the task into clear, self-contained sub-tasks — each sub-task must be solvable independently
4. Each agent may appear at most once
5. If the task is simple and general, use only the "general" agent

IMPORTANT: Return ONLY valid JSON. No explanation. No markdown fences.

Format:
{{"tasks": [{{"agent": "agent_name", "subtask": "specific sub-task description"}}]}}
""")


async def route_task(task: str, agent_registry: dict) -> list[dict]:
    """
    Ask the orchestrator LLM to break the task into per-agent sub-tasks.

    Args:
        task:           Original user task string.
        agent_registry: Dict of {name: agent} — used to build the description.

    Returns:
        List of {"agent": str, "subtask": str} dicts.
        Falls back to [{"agent": "general", "subtask": task}] on parse failure.
    """
    agents_desc = "\n".join(
        f"  - {name}: {agent.description}"
        for name, agent in agent_registry.items()
    )

    chain = _ROUTER_PROMPT | ORCHESTRATOR_LLM | StrOutputParser()
    raw = ""
    thinking_buf = ""   
    async for chunk in chain.astream({"task": task, "agents_desc": agents_desc}):
        raw += chunk
        thinking_buf = (thinking_buf + chunk)[-55:]                
        sys.stdout.write(f"\r  📋 Thinking: {thinking_buf:<55}")   
        sys.stdout.flush()                                          

    sys.stdout.write("\r" + " " * 72 + "\r")                       
    sys.stdout.flush()    

    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:-1]).strip()

    try:
        data  = json.loads(clean)
        tasks = data.get("tasks", [])

        # Validate every named agent actually exists
        valid = [t for t in tasks if t.get("agent") in agent_registry]
        if not valid:
            raise ValueError("No valid agents found in plan")
        return valid

    except Exception as exc:
        print(f"[ROUTER] Parse error ({exc}) — falling back to general agent")
        return [{"agent": "general", "subtask": task}]
