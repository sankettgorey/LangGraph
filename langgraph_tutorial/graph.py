from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from dotenv import load_dotenv



# ---------------------------
# ENV
# ---------------------------
load_dotenv("./langgraph_tutorial/.env")

# ---------------------------
# LLMs
# ---------------------------
generator_llm = ChatOllama(model="qwen3:8b")
feedback_llm = ChatOllama(model="qwen3:8b")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    count: Optional[int]
    final_answer: Optional[str]


# ---------------------------
# Generation Node
# ---------------------------
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in answering the users question. "
            "Answer each question in detail as if you are explaining it to a 6th grader."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_chain = generation_prompt | generator_llm


def llm_node(state: AgentState):
    result = generation_chain.invoke(state["messages"])
    return {
        "messages": [AIMessage(content=result.content)],
        "final_answer": result.content,
    }


# ---------------------------
# Feedback Node
# ---------------------------
feedback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert critique who constructively criticizes the answer.\n"
            "Give feedback in 2–3 lines based on:\n"
            "1. Quality\n2. Depth\n3. Completeness",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

feedback_chain = feedback_prompt | feedback_llm


def feedback_node(state: AgentState):
    output = feedback_chain.invoke(state["messages"])
    return {
        "messages": state["messages"] + [HumanMessage(content=output.content)],
        "count": state.get('count', 0) + 1,
    }


def shall_continue(state: AgentState):
    if state.get('count', 0) > 1:
        return END
    return "feedback"


# ---------------------------
# Graph
# ---------------------------
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("feedback", feedback_node)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    shall_continue,
    {"feedback": "feedback", END: END},
)


conn = sqlite3.connect(
    database = 'chatbot_with_reflection.db',
    check_same_thread=False
)

checkpointer = SqliteSaver(conn = conn)


workflow = graph.compile(checkpointer = checkpointer)


initial_state: AgentState = {
    "messages": [HumanMessage(content = 'can you tell me whats my name?')]
}

config = {"configurable": {"thread_id": "1"}}


output = workflow.invoke(initial_state, config = config)

print(output['final_answer'])