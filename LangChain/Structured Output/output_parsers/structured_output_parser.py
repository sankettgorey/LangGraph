from langchain_ollama import ChatOllama
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_core.prompts import PromptTemplate


model = ChatOllama(model = 'llama3.2:3b')

schema = [
    ResponseSchema(name = 'fact_1', description='fact 1 about the topic'),
    ResponseSchema(name = 'fact_2', description='fact 2 about the topic'),
    # ResponseSchema(name = 'fact_3', description='fact 3 about the topic'),
    # ResponseSchema(name = 'fact_4', description='fact 4 about the topic'),
]


parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template = 'give me 4 facts about the {topic} \n {format_instructions}',
    input_variables= ['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)


chain = template | model | parser

output = chain.invoke({'topic': 'black hole'})

prompt = template.invoke({'topic': 'black hole'})

output = model.invoke(prompt)

parsed_output = parser.parse(output.content)
print(parsed_output)