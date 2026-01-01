from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
import operator

from langgraph.checkpoint.memory import InMemorySaver
import sqlite3

# -----------------------------
# 1. Correct AgentState
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# -----------------------------
# 2. Proper Tool Definition
# -----------------------------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """multiplies two numbers together."""
    return a * b

tools = [add_numbers, multiply]

# -----------------------------
# 3. LLM with Native Tool Binding
# -----------------------------
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# 4. Nodes (Simplified)
# -----------------------------
def llm_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Use prebuilt ToolNode - handles parsing automatically
tool_node = ToolNode(tools)

# -----------------------------
# 5. Build Graph (Standard Pattern)
# -----------------------------
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")

checkpiont = InMemorySaver()

app = graph.compile(checkpointer=checkpiont)

# -----------------------------
# 6. Run
# -----------------------------


while True:
    usuer_input = input("Enter Question: ")
    result = app.invoke({
    "messages": [HumanMessage(content=usuer_input)],
    },
    config={"configurable": {"thread_id": '1'}}

)

    print("FINAL ANSWER:", result["messages"][-1].content)
    print()
