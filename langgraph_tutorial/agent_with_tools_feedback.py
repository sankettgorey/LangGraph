from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# ---------------------------
# ENV
# ---------------------------
load_dotenv("./langgraph_tutorial/.env")

# ---------------------------
# LLM
# ---------------------------
llm = ChatOllama(model="qwen2.5:7b-instruct")

# ---------------------------
# TOOLS
# ---------------------------
search = TavilySearch(search_results=3)

@tool
def addition(a: int, b: int):
    """Add two integers and return the result."""
    return a + b


@tool
def division(a: int, b: int):
    """Divide a by b and return the result. b must be non-zero."""
    if b <= 0:
        return "Division by zero"
    return a / b


TOOLS = [addition, division, search]
tools_node = ToolNode(TOOLS)
llm_with_tools = llm.bind_tools(TOOLS)

# ---------------------------
# STATE TYPE
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    count: int

# ---------------------------
# NODES
# ---------------------------
def generation_node(state: AgentState):
    """
    Call the LLM (with tools bound). Return the raw LLM response as the single-message list.
    """
    response = llm_with_tools.invoke(state["messages"])
    # return the actual model message (keeps tool_calls metadata)
    return {"messages": [response]}

def critique_node(state: AgentState):
    """
    Ask the LLM (without tools) to critique the last answer.
    We prepend a short instruction but preserve history by passing state["messages"].
    """
    
    critique = llm.invoke(
        [HumanMessage(content="Please critique the last answer briefly and point out factual errors.")] + state["messages"]
    )
    return {
        "messages": [HumanMessage(content=critique.content)],
        "count": state["count"] + 1
    }

def modification_node(state: AgentState):
    """
    Ask the LLM-with-tools to improve the last answer based on critique.
    """
    improved = llm_with_tools.invoke(
        [HumanMessage(content="Improve the previous answer using the critique.")] + state["messages"]
    )
    return {"messages": [improved]}

# ---------------------------
# ROUTERS (ONE conditional per node)
# ---------------------------
def generation_router(state: AgentState):
    """
    Decide where to go after generation.
    - If the last model message requested tools -> go to 'tools'
    - Otherwise -> go to 'critique'
    This router only returns labels we map below (no END)
    """

    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "critique"

def modification_router(state: AgentState):
    """
    Decide where to go after modification:
    - If last message contains tool_calls -> 'tools'
    - Else if we've reached the reflection limit -> END
    - Else -> 'critique'
    This single router handles both tool routing and the end condition.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if state["count"] >= 2:
        return END
    return "critique"

# ---------------------------
# BUILD GRAPH
# ---------------------------
graph = StateGraph(AgentState)

graph.add_node("generation", generation_node)
graph.add_node("tools", tools_node)
graph.add_node("critique", critique_node)
graph.add_node("modification", modification_node)

# entry
graph.add_edge(START, "generation")

# generation -> tools OR critique  (only one conditional router on 'generation')
graph.add_conditional_edges(
    "generation",
    generation_router,
    {
        "tools": "tools",
        "critique": "critique"
    }
)

# tools -> generation (tools results go back to generation)
graph.add_edge("tools", "generation")

# critique -> modification
graph.add_edge("critique", "modification")

# modification -> (single conditional router) -> tools | critique | END
graph.add_conditional_edges(
    "modification",
    modification_router,
    {
        "tools": "tools",
        "critique": "critique",
        END: END
    }
)

workflow = graph.compile()

# ---------------------------
# RUN
# ---------------------------
result = workflow.invoke(
    {
        "messages": [HumanMessage(content="What's the latest update on tariffs on India?")],
        "count": 0
    }
)

print("\nFINAL ANSWER:\n")
# the last message in messages list should be an AIMessage/HumanMessage object
print(result["messages"][-1].content)
