from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from schema import AnswerQuestion, RevisorTemplate

from dotenv import load_dotenv

load_dotenv('../.env')


llm=ChatOllama(model='qwen3:8b')

# llm=ChatOpenAI(model='gpt-5-nano')


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an exper AI researcher.

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries separately** for
            researching improvements. Do not include them inside the reflection.
            """
        ),
        MessagesPlaceholder(variable_name='messages'),
        (
            "system",
            "Answer the user's quesion above using the required format"
        )
    ]
)

actor_prompt_template_partial = actor_prompt_template.partial(first_instruction='write a detailed ~250 words essay about the given topic.')


actor_chain = actor_prompt_template_partial | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion')

# output = actor_chain.invoke(input={'messages': [HumanMessage(content='write ~250 words essay about the evolution of AI ')]})
# print(output)


# ========================================== REVISOR CHAIN ==========================================
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(tools=[RevisorTemplate], tool_choice='RevisorTemplate')


