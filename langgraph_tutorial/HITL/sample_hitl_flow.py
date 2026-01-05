from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama

from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from typing import TypedDict, List, Annotated, Optional


llm = ChatOllama(model = 'qwen3:8b')


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    decision = interrupt({
        "type": "Approval or reject\n",
        "reason": "Model is about to answer the user's question\n",
        "question": f"{state["messages"][-1].content}\n",
        "instruction": "Approve this question: [yes/no]"
    })

    if decision["approved"] == "no":
        return {
            "messages": [AIMessage(content = "Not Approved")]
        }
    
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


graph = StateGraph(ChatState)

graph.add_node('chat', chat_node)

graph.set_entry_point('chat')
graph.add_edge('chat', END)

checkpointer = InMemorySaver()

config = {'configurable': {'thread_id': '1'}}

workflow = graph.compile(checkpointer=checkpointer)

initial_state: ChatState = {
    "messages": [HumanMessage(content = 'tell me a joke')]
}

result = workflow.invoke(initial_state, config=config)
# print(result)
# print()

# print(result["__interrupt__"][0].value)
# print()


user_input = input(f"\nBackend Message:\n{result['__interrupt__'][0].value}: ")

# print(user_input)

final_output = workflow.invoke(
    Command(
        resume={
            "approved": user_input
        }
    ),
    config=config
)

print(final_output["messages"][-1].content)