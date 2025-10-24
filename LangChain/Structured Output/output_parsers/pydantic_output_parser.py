from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field


model = ChatOllama(model = 'mistral:7b-instruct')

# model = ChatOpenAI(api_key='sk-rzaTtnp_8KEw_c4qvbkZMotzUOb-1AJ2I4h88XxFC4T3BlbkFJW_LYKAQ-NEvh3jCdphCLh2kqnu2mBkmnF9NwZv2rUA',
#                    model = 'gpt-4o')



class Person(BaseModel):

    name: str = Field(description='name of a person')
    age: int = Field(description='age of a person',)
    city: str = Field(description='place where the person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template = 'Give me the name, age and city of {place} person \n{format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

try:
    output = chain.invoke({'place': 'australian'})
    print(output)

except Exception as e:
    print(e)