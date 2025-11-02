from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama
from langchain_community.tools import TavilySearchResults

from langchain_core.tools import tool
from langchain.agents import create_react_agent


from langchain import hub


from  dotenv import load_dotenv
import datetime

load_dotenv('../.env')

llm = ChatOllama(model = 'qwen3:8b')

search_tool = TavilySearchResults(search_depth = 'basic')


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%m%s"):
    """returns the current date and time of the system in the specified format"""

    current_time = datetime.datetime.now()

    format_time = current_time.strftime(format)
    print(format_time)

    return format_time

tools = [search_tool, get_system_time]


react_prompt = hub.pull('hwchase17/react')



react_agent_runnable = create_react_agent(
    prompt = react_prompt,
    tools = tools,
    llm = llm,
    # verbose=True
)