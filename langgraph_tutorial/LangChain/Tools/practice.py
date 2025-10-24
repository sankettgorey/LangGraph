import threading

from langchain_community.tools import tool, StructuredTool, DuckDuckGoSearchResults

from langchain_ollama import ChatOllama

from pydantic import Field, BaseModel

model = ChatOllama(model = 'mistral:7b-instruct')


class Multiply(BaseModel):
    a: float = Field(description='first number to multiply')
    b: float =Field(description='second number to multiply')


def multiply(a, b):
    '''multiplies two numbers'''
    return a * b


multiply_tool = StructuredTool.from_function(
    func = multiply,
    description= 'multiplies two numbers',
    name = 'multiply',
    args_schema=Multiply
)


search = DuckDuckGoSearchResults()


structured_model = model.bind_tools(tools = [multiply_tool, search])


output = structured_model.invoke(input='who won asia cup?')
print(output)