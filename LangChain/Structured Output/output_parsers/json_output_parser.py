from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model = 'llama3.2:3b')

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me 5 facts about {topic}.\n {format_instructions}',
    partial_variables={'format_instructions': parser.get_format_instructions()},
    input_variables=['topic']
)

chain = template | model | parser

output = chain.invoke({'topic': 'supply chain'})
print(output)
