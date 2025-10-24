from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults

from dotenv import load_dotenv

from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(model = 'mistral:7b-instruct')


search = TavilySearchResults(search_depth='basic')


agent = initialize_agent(
    tools = [search], 
    llm=llm,
    verbose=True,
    agent='zero-shot-react-description'
)

result=agent.invoke('whats the weather in pune today?')