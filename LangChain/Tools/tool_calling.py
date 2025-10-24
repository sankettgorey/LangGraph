from langchain_core.tools import tool, Tool, BaseTool

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate, AIMessagePromptTemplate


llm = ChatOllama(model = "mistral:7b-instruct")
# llm = ChatOllama(model = 'qwen2.5:14b')


@tool
def multiply(a: int, b: int):
    '''multiplies two numbers'''

    return f'Output of equation {a} * {b} is {a * b}'



llm_with_tools = llm.bind_tools(tools=[multiply])



prompt = PromptTemplate(
    template='{equation}',
    input_variables=['equation'],
)

parser = StrOutputParser()


chain = prompt | llm_with_tools

output = chain.invoke(
    {
        'equation': "whats the output of 4 and 5"
    }
)

print(output.tool_calls[0])
print()

result = multiply.invoke(output.tool_calls[0])

print(result)