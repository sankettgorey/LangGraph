from langchain import hub

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor

from langchain.tools import tool

from langchain_ollama import ChatOllama


import requests

model = ChatOllama(model = 'qwen2.5:7b-instruct')

# model = ChatOpenAI(api_key= "sk-rzaTtnp_8KEw_c4qvbkZMotzUOb-1AJ2I4h88XxFC4T3BlbkFJW_LYKAQ-NEvh3jCdphCLh2kqnu2mBkmnF9NwZv2rUA",
#                    model = 'gpt-4o')


search = DuckDuckGoSearchRun()

prompt = hub.pull('hwchase17/react')

@tool
def get_weather(city: str):
    '''
    returns the current weather of the given city
    '''
    
    url = f"https://api.weatherstack.com/current?access_key={key}&query={city}"

    response = requests.get(url)
    return response.json()


agent = create_react_agent(
    tools=[search, get_weather],
    llm = model,
    prompt=prompt
)


executor = AgentExecutor(
    agent = agent,
    verbose=True,
    tools=[search, get_weather]
)

output = executor.invoke(
    input = {
        'input': 'what is the current weather of capital of madhya pradesh'
    }
)

print(output['output'])