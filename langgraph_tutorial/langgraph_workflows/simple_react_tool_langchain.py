from langchain_ollama import ChatOllama
from langchain_community.tools import TavilySearchResults, tool, StructuredTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pydantic import Field, BaseModel

load_dotenv()


@tool
def get_system_datetime(today: str) -> str:

    """Returns the today's date and current time of the system in specified format."""
    
    format: str = "%Y-%m-%d %H:%M:%S"
    return datetime.now().strftime(format)


search_tool = TavilySearchResults(search_depth='basic')
tools = [search_tool, get_system_datetime]


# llm = ChatOllama(model="mistral:7b-instruct")
llm = ChatOllama(model="qwen2.5:14b")
# llm = ChatOpenAI()


prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

result = executor.invoke({"input": "whats the temperature of pune now?"})
print(result["output"])
