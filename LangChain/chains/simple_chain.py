from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser


try:
    prompt = PromptTemplate(
        template = 'give me 5 interesting facts about the topic: {topic}',
        input_variables=['topic']
    )

    model = ChatOllama(model = 'llama3.2:3b')

    parser = StrOutputParser()

    chain = prompt | model | parser

    print(chain.get_graph().draw_ascii())
    print(chain.get_graph().print_ascii())

    output = chain.invoke(
        {
            'topic': 'supply chain'
        }
    )
    print(output)

except Exception as e:
    print(str(e))