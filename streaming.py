import sys
import asyncio
from langchain_core.messages import HumanMessage

# ANSI escape codes — work on any modern terminal (Linux, macOS, Windows 10+)
_DIM    = "\033[2m"     # dim / grey
_RESET  = "\033[0m"     # reset all attributes
_CLEAR  = "\r\033[K"    # carriage-return + erase line


class StreamRenderer:
    """
    Stateful renderer that handles the think→answer transition.

    States:
      idle       → waiting for first event
      thinking   → receiving reasoning_content tokens
      answering  → receiving content tokens (reasoning cleared)
    """

    def __init__(self, agent_name: str):
        self.agent_name     = agent_name
        self._state         = "idle"
        self._thinking_buf  = ""   # full accumulated thinking text (for clearing)
        self._think_chars   = 0    # chars printed on current thinking line

    def _print_thinking_token(self, token: str) -> None:
        """
        Stream reasoning tokens in-place on one line.
        We keep a rolling display: 'Thinking: <last 72 chars>…'
        """
        self._thinking_buf += token

        # Build the display line — show only the tail so it fits one line
        tail = self._thinking_buf.replace("\n", " ")
        if len(tail) > 72:
            tail = "…" + tail[-72:]

        line = f"{_DIM}Thinking: {tail}{_RESET}"
        sys.stdout.write(f"{_CLEAR}{line}")
        sys.stdout.flush()
        self._think_chars = len(line)

    def _clear_thinking_line(self) -> None:
        """Erase the thinking line completely before streaming the answer."""
        if self._think_chars > 0:
            sys.stdout.write(_CLEAR)
            sys.stdout.flush()
            self._think_chars = 0

    def on_thinking_token(self, token: str) -> None:
        if not token:
            return
        if self._state == "idle":
            self._state = "thinking"
        self._print_thinking_token(token)

    def on_answer_token(self, token: str) -> None:
        if not token:
            return
        if self._state in ("idle", "thinking"):
            # First answer token — clear the thinking line and print header
            self._clear_thinking_line()
            label = f"\n{self.agent_name.upper()}: " if self.agent_name else "\n"
            sys.stdout.write(label)
            sys.stdout.flush()
            self._state = "answering"
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_tool_start(self, tool_name: str) -> None:
        self._clear_thinking_line()
        sys.stdout.write(f"\n  🔧 {tool_name}…")
        sys.stdout.flush()

    def on_tool_end(self, tool_name: str) -> None:
        sys.stdout.write(f" ✓\n")
        sys.stdout.flush()

    def finish(self) -> None:
        """Called after the last event — ensure we end on a clean line."""
        self._clear_thinking_line()
        if self._state == "answering":
            sys.stdout.write("\n")
            sys.stdout.flush()


async def stream_agent_response(
    runnable,
    subtask: str,
    agent_name: str = "",
    node_filter: str | None = None,
) -> str:
    """
    Stream a single agent's response with live thinking display.

    Args:
        runnable:    The agent runnable (from agent.get_runnable())
        subtask:     The sub-task string to send to the agent
        agent_name:  Display name shown before the answer
        node_filter: If set, only stream tokens from this LangGraph node name.
                     Use None to stream from all nodes (plain LLM chains).

    Returns:
        The complete answer text (accumulated from all answer tokens).
    """
    renderer     = StreamRenderer(agent_name)
    answer_parts = []

    # Determine input format — ReAct agents expect {"messages": [...]}
    # Plain LLM chains expect {"subtask": "..."}
    if hasattr(runnable, "nodes"):
        # It's a compiled LangGraph (ReAct agent from create_react_agent)
        inp = {"messages": [HumanMessage(content=subtask)]}
    else:
        inp = {"subtask": subtask}

    async for event in runnable.astream_events(inp, version="v2"):
        kind = event.get("event", "")
        meta = event.get("metadata", {})
        node = meta.get("langgraph_node", "")

        # ── Filter by node if requested ──────────────────────────────────────
        if node_filter and node and node != node_filter:
            continue

        # ── LLM token chunk ──────────────────────────────────────────────────
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue

            # Thinking token (qwen3 reasoning=True → additional_kwargs)
            thinking = chunk.additional_kwargs.get("reasoning_content", "")
            if thinking:
                renderer.on_thinking_token(thinking)

            # Answer token
            answer = chunk.content or ""
            if answer:
                renderer.on_answer_token(answer)
                answer_parts.append(answer)

        # ── Tool lifecycle ───────────────────────────────────────────────────
        elif kind == "on_tool_start":
            renderer.on_tool_start(event.get("name", "tool"))

        elif kind == "on_tool_end":
            renderer.on_tool_end(event.get("name", "tool"))

    renderer.finish()
    return "".join(answer_parts)


async def stream_graph_synthesis(graph, initial_state, config=None) -> dict:
    """
    Run the full Whatson graph with streaming for the synthesize node.

    During plan/approval/dispatch: runs silently (agents print their own output).
    During synthesize: streams tokens live with thinking display.

    Returns the final state dict.
    """
    renderer     = StreamRenderer("Whatson")
    answer_parts = []
    final_state  = {}

    # Track whether we've entered the synthesize node
    in_synthesize = False

    async for event in graph.astream_events(initial_state, version="v2", config=config):
        kind = event.get("event", "")
        meta = event.get("metadata", {})
        node = meta.get("langgraph_node", "")

        # ── Synthesize node: stream tokens live ──────────────────────────────
        if node == "synthesize" and kind == "on_chat_model_stream":
            if not in_synthesize:
                in_synthesize = True
                print("\n" + "═" * 60)
                print("WHATSON:")
                print("═" * 60)

            chunk = event["data"].get("chunk")
            if chunk is None:
                continue

            thinking = chunk.additional_kwargs.get("reasoning_content", "")
            if thinking:
                renderer.on_thinking_token(thinking)

            answer = chunk.content or ""
            if answer:
                renderer.on_answer_token(answer)
                answer_parts.append(answer)

        # ── Capture final state from chain_end on the top-level graph ────────
        elif kind == "on_chain_end" and not node:
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                final_state.update(output)

    renderer.finish()

    # Patch the streamed answer into final_state
    if answer_parts:
        final_state["final_answer"] = "".join(answer_parts)

    return final_state
