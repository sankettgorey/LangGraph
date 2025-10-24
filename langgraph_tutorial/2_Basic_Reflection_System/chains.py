from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama



generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a twitter techie influencer assistant tasked with writing excellent twitter post."
            " Generate best twitter post possible for the user's request."
            " If user provides critique, respond with the revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

reflector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate grading out of 10 and recommendations for user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style etc."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)


llm=ChatOllama(model='qwen2.5:7b-instruct')


generation_chain = generation_prompt | llm

relfection_chain = reflector_prompt | llm