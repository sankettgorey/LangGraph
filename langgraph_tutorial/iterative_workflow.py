from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

from pydantic import BaseModel, Field
from typing import Annotated, Literal, List, Dict, Optional, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


generator_llm = ChatOllama(model = 'qwen2.5:7b-instruct')
evaluator_llm = ChatOllama(model = 'qwen2.5:7b-instruct')
optimizer_llm = ChatOllama(model = 'qwen2.5:7b-instruct')



class TweetState(TypedDict):

    topic: str
    tweet: str
    evaluation: Literal['approval', 'needs_improvement']
    feedback: str
    iteration: int
    max_iteration: int

    tweet_history: Annotated[List[BaseMessage], add_messages]
    feedback_history: Annotated[List[BaseMessage], add_messages]


def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    result = generator_llm.invoke(messages)

    return {
        'tweet': result.content,
        'tweet_history': [result]
    }


class TweetEvaluation(BaseModel):

    evaluation: Literal['approved', 'needs_improvement'] = Field(..., description='final evaluation result')
    feedback: str = Field(..., description='feedback to improve the tweet')

evaluator_model = evaluator_llm.with_structured_output(TweetEvaluation)

def evaluate_tweet(state: TweetState):

    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
]
    output = evaluator_model.invoke(messages)

    return {
        'evaluation': output.evaluation,
        'feedback': output.feedback,
        'feedback_history': [output.feedback]
    }


def optimize_tweet(state: TweetState):

    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}



def route_evaluation(state: TweetState):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'
    

graph = StateGraph(TweetState)

graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

graph.add_conditional_edges('evaluate', route_evaluation, {'approved': END, 'needs_improvement': 'optimize'})
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()

initial_state = {
    "topic": "vastness of space and AI",
    "iteration": 1,
    "max_iteration": 5
}
result = workflow.invoke(initial_state)
print(result)

# sample outout
x = {'topic': 'vastness of space and AI', 
     'tweet': "🚀 In the vastness of space, AI is like that friend who always has to add a filter to every pic and then tells you it's the view from their spaceship. #SpaceAI #FilterLife 🌌📸", 
     'evaluation': 'approved', 
     'feedback': "This tweet scores well across all criteria. It is original by comparing AI to a friend who overuses filters, providing an interesting analogy that hasn't been overused. The humor is subtle but effective, creating a relatable scenario for many readers. The punchiness is on point with concise language and vivid imagery. There's also high virality potential as the concept of space and AI resonates with tech enthusiasts and casual followers alike. Finally, it adheres to proper tweet format without exceeding character limits or ending in a deflating line.", 
     'iteration': 1, 
     'max_iteration': 5, 
     'tweet_history': [AIMessage(content="🚀 In the vastness of space, AI is like that friend who always has to add a filter to every pic and then tells you it's the view from their spaceship. #SpaceAI #FilterLife 🌌📸", additional_kwargs={}, response_metadata={'model': 'qwen2.5:7b-instruct', 'created_at': '2025-12-17T13:57:11.9021957Z', 'done': True, 'done_reason': 'stop', 'total_duration': 1568879100, 'load_duration': 226087500, 'prompt_eval_count': 101, 'prompt_eval_duration': 395832100, 'eval_count': 46, 'eval_duration': 727015000, 'logprobs': None, 'model_name': 'qwen2.5:7b-instruct', 'model_provider': 'ollama'}, id='lc_run--019b2c99-b83b-7a91-8de8-d513884735eb-0', usage_metadata={'input_tokens': 101, 'output_tokens': 46, 'total_tokens': 147})], 'feedback_history': [HumanMessage(content="This tweet scores well across all criteria. It is original by comparing AI to a friend who overuses filters, providing an interesting analogy that hasn't been overused. The humor is subtle but effective, creating a relatable scenario for many readers. The punchiness is on point with concise language and vivid imagery. There's also high virality potential as the concept of space and AI resonates with tech enthusiasts and casual followers alike. Finally, it adheres to proper tweet format without exceeding character limits or ending in a deflating line.", additional_kwargs={}, response_metadata={}, id='35af712d-26e7-4a11-b430-e7aa60c44cf4')]}