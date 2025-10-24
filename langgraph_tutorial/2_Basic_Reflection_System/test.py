from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

load_dotenv()

# --------------------------------------------------------------------
# 1️⃣ Prompts and Chains
# --------------------------------------------------------------------

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter tech influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request. "
            "If the user provides critique, respond with the revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate grading and recommendations for user's tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOllama(model="qwen2.5:7b-instruct")

generation_chain = generation_prompt | llm
reflection_chain = reflector_prompt | llm

# --------------------------------------------------------------------
# 2️⃣ Define the State
# --------------------------------------------------------------------

from typing import TypedDict

class TweetState(TypedDict):
    messages: List[BaseMessage]


# --------------------------------------------------------------------
# 3️⃣ Define the Node Functions
# --------------------------------------------------------------------

def generate_node(state: TweetState):
    """Generate tweet based on the messages"""
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [response]}


def reflect_node(state: TweetState):
    """Reflect and critique the generated tweet"""
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [HumanMessage(content=response.content)]}


# --------------------------------------------------------------------
# 4️⃣ Build the Graph using StateGraph (✅ modern replacement)
# --------------------------------------------------------------------

REFLECT = "reflect"
GENERATE = "generate"

graph = StateGraph(TweetState)

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)


# --------------------------------------------------------------------
# 5️⃣ Conditional Edge Logic
# --------------------------------------------------------------------

def should_continue(state: TweetState):
    """Conditionally decide if the loop should end or reflect again"""
    if len(state["messages"]) > 6:
        return END
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)


# --------------------------------------------------------------------
# 6️⃣ Compile and Run
# --------------------------------------------------------------------

app = graph.compile()

# Visualize flow
# print(app.get_graph().draw_mermaid())
print(app.get_graph().print_ascii())

# Run
# response = app.invoke({"messages": [HumanMessage(content="AI Agents taking over content creation")]})

# print("\nFinal Output:\n", response["messages"][-1].content)
