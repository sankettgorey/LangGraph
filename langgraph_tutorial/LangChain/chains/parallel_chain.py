from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama


parser = StrOutputParser()

model1 = ChatOllama(model = 'mistral:7b-instruct')
model2 = ChatOllama(model = 'mistral:7b-instruct')


try:
    prompt1 = PromptTemplate(
        template='generate detailed report on the topic: {topic}',
        input_variables=['topic']
    )


    prompt2 = PromptTemplate(
        template = 'generate 5 question quiz on the following topic: {topic}',
        input_variables=['topic']
    )

    prompt3 = PromptTemplate(
        template = 'Merge the provided report and questions into single document. \ntext -> {text}\n\nquestions:  -> {questions}',
        input_variables=['text', 'questions']
    )

    parallel_chain = RunnableParallel(
        {
            'text': prompt1 | model1 | parser,
            'questions': prompt2 | model2 | parser
        }
    )

    seq_chain = prompt3 | model1 | parser

    merged_chain = parallel_chain | seq_chain

    output = merged_chain.invoke(
        {
            'topic': 'architecture of mistral model'
        }
    )

    print(output)


except Exception as e:
    print(str(e))