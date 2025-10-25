# tools_execution.py
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
# from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch


from typing import List, TypedDict
import json

from dotenv import load_dotenv

load_dotenv('../.env')



class ReflexionState(TypedDict):
    messages: List[BaseMessage]
    iteration: int


tavily_tool = TavilySearch(max_results=5)

def execute_tools(state: ReflexionState) -> ReflexionState:
    print('inside execute tools')
    # print(state['messages'])
    # print(state)

    last_ai_message: AIMessage = {[state['messages'][-1].content][0]}
    # print(last_ai_message)
    # print(type(last_ai_message))

    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        # ✅ Must return dict, even if no new tool messages
        print('nothing as tool message')
        return {"messages": state["messages"]}

    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call['name'] in ['AnswerQuestion', 'ReviseAnswer']:
            call_id = tool_call['id']
            search_queries = tool_call['args'].get('search_queries', [])

            query_result = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_result[query] = result

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_result),
                    tool_call_id=call_id
                )
            )
    print(f'Tool messaage: {state["messages"] + tool_messages}')
    # ✅ Merge old + new messages into the state dict
    return {"messages": state["messages"] + tool_messages}
###############




