from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.graph import START, END, StateGraph

from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv


load_dotenv()


llm = ChatOllama(model = "qwen3:8b")


loader = PyPDFLoader("./intro-to-ml.pdf")

docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 500)
document = splitter.split_documents(docs)
# print(len(document))



embeddings=OllamaEmbeddings(model = "nomic-embed-text:latest")

vector_store = FAISS.from_documents(documents = document, embedding=embeddings)

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {'k': 5})



@tool
def rag_tool(query: str):
    """
    Args:
        query (str): _description_
    Return the relevant information from the pdf document.
    use this tool when user asks factual or conceptual questions about the pdf
    """

    result = retriever.invoke(query)

    # print('retrieved result')

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    # print(context)
    return {
        "query": query,
        "content": context,
        "metadata": metadata
    }

tools = [rag_tool]
tools_node = ToolNode(tools=tools)


llm_with_tools = llm.bind_tools(tools)


class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: RAGState):
    print('in chat node')
    output = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [output]
    }


graph = StateGraph(RAGState)
graph.add_node("chat_node", chat_node)
graph.add_node('tools', tools_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)


while True:
    user_input = input("Enter Question: ")

    if user_input.lower() in ['quit', 'stop', 'break', 'exit']:
        break

    initial_state: RAGState = {"messages": [HumanMessage(content = user_input)],}

    results = chatbot.invoke(initial_state, config = {
                                                       "configurable": {"thread_id":"1"}
                                                     })

    print(results["messages"][-1].content)
