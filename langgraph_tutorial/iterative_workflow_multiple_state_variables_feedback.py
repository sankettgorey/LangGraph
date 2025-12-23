from typing import Annotated, TypedDict, Optional, List, Literal
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_tavily import TavilySearch

from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph

from pydantic import BaseModel, Field

# =================================================
# ENV
# =================================================
load_dotenv("./langgraph_tutorial/.env")

# =================================================
# CONSTANTS
# =================================================
MAX_ITERS = 2

# =================================================
# LLMs
# =================================================
generation_llm = ChatOllama(model="qwen3:8b")
feedback_llm = ChatOllama(model="qwen3:8b")

# =================================================
# TOOLS
# =================================================
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


search = TavilySearch(max_results=3)
tools = [multiply, search]
tools_node = ToolNode(tools=tools)

generation_llm_with_tools = generation_llm.bind_tools(tools)

# =================================================
# STATE
# =================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    final_answer: Optional[str]
    feedback: Optional[str]
    evaluation: Optional[str]
    count: int

# =================================================
# SINGLE GENERATION NODE (SMART)
# =================================================
def generation_node(state: AgentState):
    # Detect whether tools have already been executed
    tools_already_used = any(
        isinstance(msg, ToolMessage) for msg in state["messages"]
    )

    if tools_already_used:
        # AFTER tools: force final answer, no tools
        messages = [
            SystemMessage(
                content=(
                    "You have already received tool results. "
                    "Answer the question using those results. "
                    "DO NOT call any tools again."
                )
            ),
            *state["messages"],
        ]

        output = generation_llm.invoke(messages)

    else:
        # FIRST pass: tools allowed
        messages = [
            SystemMessage(
                content=(
                    "You are an expert assistant. Answer the question clearly. "
                    "Use tools ONLY if required to fetch facts."
                )
            ),
            HumanMessage(content=f"Question: {state['question']}"),
        ]

        output = generation_llm_with_tools.invoke(messages)

    return {
        "messages": [output],
        "final_answer": output.content,
        "count": state["count"],
    }

# =================================================
# EVALUATION (STRUCTURED)
# =================================================
class EvalSchema(BaseModel):
    feedback: str = Field(..., description="4-5 lines of constructive feedback")
    evaluation: Literal["APPROVE", "NEEDS IMPROVEMENT"]


eval_llm = feedback_llm.with_structured_output(EvalSchema)


def eval_node(state: AgentState):
    messages = [
        SystemMessage(
            content=(
                "You are an evaluator. Judge the answer based on "
                "accuracy, depth, completeness, and tone."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Answer:\n{state['final_answer']}"
            )
        ),
    ]

    output: EvalSchema = eval_llm.invoke(messages)

    new_count = state["count"] + 1

    print(
        f"[EVAL] iteration={new_count} "
        f"decision={output.evaluation}"
    )

    return {
        "feedback": output.feedback,
        "evaluation": output.evaluation,
        "count": new_count,
    }

# =================================================
# OPTIMIZATION (NO TOOLS)
# =================================================
def optimize_node(state: AgentState):
    messages = [
        SystemMessage(
            content=(
                "Improve the answer strictly using the feedback. "
                "Do NOT introduce new facts or tools."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Original Answer:\n{state['final_answer']}\n\n"
                f"Feedback:\n{state['feedback']}"
            )
        ),
    ]

    output = generation_llm.invoke(messages)

    return {
        "messages": [output],
        "final_answer": output.content,
        "count": state["count"],
    }

# =================================================
# ROUTING
# =================================================
def generation_condition(state: AgentState):
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "eval"


def eval_condition(state: AgentState):
    if state["evaluation"] == "APPROVE":
        return END

    if state["count"] >= MAX_ITERS:
        print("⚠️ Max iterations reached — forcing END")
        return END

    return "optimize"

# =================================================
# GRAPH
# =================================================
graph = StateGraph(AgentState)

graph.add_node("generation", generation_node)
graph.add_node("tools", tools_node)
graph.add_node("eval", eval_node)
graph.add_node("optimize", optimize_node)

graph.add_edge(START, "generation")

graph.add_conditional_edges(
    "generation",
    generation_condition,
    {
        "tools": "tools",
        "eval": "eval",
    },
)

graph.add_edge("tools", "generation")

graph.add_conditional_edges(
    "eval",
    eval_condition,
    {
        "optimize": "optimize",
        END: END,
    },
)

graph.add_edge("optimize", "eval")

workflow = graph.compile()

# =================================================
# RUN
# =================================================
initial_state: AgentState = {
    "question": "What's the latest news on the Epstein files?",
    "messages": [],
    "count": 0,
    "final_answer": None,
    "feedback": None,
    "evaluation": None,
}

result = workflow.invoke(initial_state)

print("\n================ FINAL ANSWER ================\n")
print(result["final_answer"])
