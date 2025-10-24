from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# using string outout parser
from langchain_core.output_parsers import StrOutputParser



llm = ChatOllama(model = 'llama3.2:3b')


template1 = PromptTemplate(
    template="Write a detailed report on the topic: {topic}",
    input_variables=['topic']
)

# prompt1 = template1.invoke({'topic': 'black hole'})

# report = llm.invoke(prompt1)


template2 = PromptTemplate(
    template = 'Write a 10 line summary on the following: \n{report}',
    input_variables=['report']
)

# prompt2 = template2.invoke({'report': report.content})

# final_output = llm.invoke(prompt2)
# print(final_output.content)



parser = StrOutputParser()


chain = template1 | llm | parser | template2 | llm | parser

output = chain.invoke(
    {
        'topic': 'role of AI in supply chain'
    }
)

print(output)