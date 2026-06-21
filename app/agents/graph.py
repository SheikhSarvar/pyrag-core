"""
LangGraph agent — T44, T49.
ReAct-style agent that uses retrieval tools to answer multi-step queries.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.logging import get_logger
from app.services.llm.base import calculate_cost

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages: list[BaseMessage]
    dataset_id: str
    iterations: int
    max_iterations: int


def build_rag_agent(
    dataset_id: str,
    provider: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
    tool_names: list[str] | None = None,
):
    """
    Build a LangGraph ReAct agent for agentic RAG over a dataset.

    The agent is given retrieval tools and iterates until it has enough
    information to produce a final answer.

    Args:
        dataset_id:     Dataset to search.
        provider:       LLM provider override.
        model:          LLM model override.
        max_iterations: Guard against infinite loops.
        tool_names:     Subset of tools to expose: ['retriever', 'search', 'knowledge', 'dataset'].
                        Defaults to all four.

    Returns:
        Compiled LangGraph StateGraph.
    """
    from app.agents.tools.retrieval_tools import (
        create_dataset_tool,
        create_knowledge_tool,
        create_retriever_tool,
        create_search_tool,
    )

    _all_tools = {
        "retriever": lambda: create_retriever_tool(dataset_id),
        "search":    lambda: create_search_tool(dataset_id),
        "dataset":   lambda: create_dataset_tool(dataset_id),
        "knowledge": lambda: create_knowledge_tool(dataset_id, provider=provider, model=model),
    }

    selected = tool_names or list(_all_tools.keys())
    tools = [_all_tools[name]() for name in selected if name in _all_tools]

    # ── LLM with tool binding ─────────────────────────────────────────────────
    def _get_llm_with_tools():
        # Build a LangChain-compatible LLM that supports tool calling.
        # We prefer OpenAI or Anthropic for tool-use quality.
        from app.core.config import get_settings
        settings = get_settings()

        if settings.openai_api_key:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model or settings.openai_default_model,
                temperature=0.1,
                api_key=settings.openai_api_key,
            )
        elif settings.anthropic_api_key:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model or settings.anthropic_default_model,
                temperature=0.1,
                api_key=settings.anthropic_api_key,
            )
        else:
            raise RuntimeError("Agent requires OpenAI or Anthropic API key for tool calling")

        return llm.bind_tools(tools)

    llm_with_tools = _get_llm_with_tools()

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def agent_node(state: AgentState) -> AgentState:
        """Call the LLM with current message history."""
        if state["iterations"] >= state["max_iterations"]:
            # Force a final answer without tool use
            final_msg = AIMessage(
                content="I have reached the maximum number of retrieval steps. "
                        "Based on what I found so far: " +
                        _extract_last_tool_output(state["messages"])
            )
            return {**state, "messages": state["messages"] + [final_msg]}

        response = llm_with_tools.invoke(state["messages"])
        return {
            **state,
            "messages": state["messages"] + [response],
            "iterations": state["iterations"] + 1,
        }

    def should_continue(state: AgentState) -> str:
        """Route: call tools or finish."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    # ── Graph assembly ────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def _extract_last_tool_output(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return str(msg.content)[:500]
    return "no tool output available."


async def run_agent(
    dataset_id: str,
    query: str,
    conversation_history: list[dict] | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    """
    Run the agentic RAG pipeline.

    Returns a dict with:
      - answer: str
      - steps: list of {step, tool, input, output}
      - messages: full message list
    """
    agent = build_rag_agent(
        dataset_id=dataset_id,
        provider=provider,
        model=model,
        max_iterations=max_iterations,
    )

    initial_messages: list[BaseMessage] = []
    for h in (conversation_history or []):
        if h["role"] == "user":
            initial_messages.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            initial_messages.append(AIMessage(content=h["content"]))
    initial_messages.append(HumanMessage(content=query))

    state = AgentState(
        messages=initial_messages,
        dataset_id=dataset_id,
        iterations=0,
        max_iterations=max_iterations,
    )

    final_state = await agent.ainvoke(state)

    # Extract final answer
    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break

    # Extract tool call steps
    steps = []
    step_num = 0
    for i, msg in enumerate(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_output = ""
                # Find the corresponding ToolMessage
                for j in range(i + 1, len(final_state["messages"])):
                    if isinstance(final_state["messages"][j], ToolMessage):
                        tool_output = str(final_state["messages"][j].content)[:300]
                        break
                steps.append({
                    "step": step_num,
                    "tool": tc["name"],
                    "input": str(tc["args"]),
                    "output": tool_output,
                })
                step_num += 1

    # Aggregate token usage across every AIMessage turn.
    # LangChain attaches `usage_metadata` (standardized field, langchain-core>=0.2)
    # to each AIMessage when the underlying provider returns usage data.
    total_input_tokens = 0
    total_output_tokens = 0
    resolved_model = model or "unknown"
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
            # response_metadata sometimes carries the resolved model name even
            # when `model` wasn't explicitly passed in (e.g. env-default model).
            meta_model = (msg.response_metadata or {}).get("model_name") or (msg.response_metadata or {}).get("model")
            if meta_model:
                resolved_model = meta_model

    total_tokens = total_input_tokens + total_output_tokens
    cost_usd = calculate_cost(resolved_model, total_input_tokens, total_output_tokens)

    # Extract sources from tool outputs. The `search` tool returns JSON with
    # structured chunk info; the `retriever`/`knowledge` tools return prose,
    # which can't be reliably parsed back into discrete sources here — those
    # steps are surfaced via `steps[].output` instead.
    sources: list[dict] = []
    for step in steps:
        if step["tool"].startswith("search_"):
            try:
                parsed = json.loads(step["output"]) if step["output"].strip().startswith("[") else []
                for item in parsed:
                    sources.append({
                        "chunk_id": item.get("id", ""),
                        "score": item.get("score", 0.0),
                        "filename": item.get("filename", ""),
                        "text": item.get("text", ""),
                    })
            except (json.JSONDecodeError, AttributeError):
                pass  # tool output was truncated or non-JSON — skip rather than guess

    return {
        "answer": answer,
        "steps": steps,
        "messages": final_state["messages"],
        "total_tokens": total_tokens,
        "prompt_tokens": total_input_tokens,
        "completion_tokens": total_output_tokens,
        "cost_usd": cost_usd,
        "model": resolved_model,
        "sources": sources,
    }
