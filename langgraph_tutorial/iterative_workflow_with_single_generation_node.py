from typing import TypedDict, List, Literal, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
load_dotenv("./langgraph_tutorial/.env")

# =====================================================
# STATE
# =====================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    reflection_count: int
    final_answer: Optional[str]

# =====================================================
# TOOLS
# =====================================================
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

search = TavilySearch(max_results=3)
tools = [multiply, search]

# =====================================================
# MODELS
# =====================================================
llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
).bind_tools(tools)

critic_llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
)

# =====================================================
# PROMPTS
# =====================================================
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict reviewer. "
     "If the answer is correct and complete, reply ONLY with 'APPROVE'. "
     "Otherwise, explain clearly what is wrong and how to fix it."),
    MessagesPlaceholder("messages")
])

generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You answer each question asked to you in 15-20 lines paragraph. "),
    MessagesPlaceholder("messages")
])

# =====================================================
# NODES
# =====================================================
def generation_node(state: AgentState):
    response = llm.invoke(state["messages"])

    final_answer = state.get("final_answer")

    # Capture final answer ONLY when model gives a real answer
    if isinstance(response, AIMessage) and not response.tool_calls:
        final_answer = response.content

    return {
        "messages": [response],
        "final_answer": final_answer,
    }

def reflection_node(state: AgentState):
    critique = critic_llm.invoke(
        reflection_prompt.format_messages(
            messages=state["messages"]
        )
    )

    count = state.get("reflection_count", 0) + 1

    # Reflection feedback injected as SYSTEM guidance
    feedback = SystemMessage(
        content=f"Reviewer feedback:\n{critique.content}"
    )

    return {
        "messages": [feedback],
        "reflection_count": count
    }

tools_node = ToolNode(tools)

# =====================================================
# DECISION LOGIC
# =====================================================
def reflection_decision(state: AgentState) -> Literal["generation", END]:
    last = state["messages"][-1]

    # Approval → stop
    if "APPROVE" in last.content:
        return END

    # Safety guard
    if state["reflection_count"] >= 3:
        return END

    return "generation"

# =====================================================
# GRAPH
# =====================================================
graph = StateGraph(AgentState)

graph.add_node("generation", generation_node)
graph.add_node("tools", tools_node)
graph.add_node("reflection", reflection_node)

graph.set_entry_point("generation")

# Tool loop (generation → tools → generation)
graph.add_conditional_edges(
    "generation",
    tools_condition
)
graph.add_edge("tools", "generation")

# Reflection gate
graph.add_edge("generation", "reflection")
graph.add_conditional_edges(
    "reflection",
    reflection_decision
)

app = graph.compile()

# =====================================================
# RUN
# =====================================================
initial_state: AgentState = {
    "messages": [
        HumanMessage(content="tell me about latest news on tariffs on India")
    ],
    "reflection_count": 0,
    "final_answer": None,
}

result = app.invoke(initial_state)

# ✅ THIS IS THE FINAL ANSWER (NOT APPROVE)
print("\nFINAL ANSWER:\n")
print(result["final_answer"])
