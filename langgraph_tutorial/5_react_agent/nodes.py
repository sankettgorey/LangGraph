from react_state import AgentState
from agent_reason_runnable import react_agent_runnable, tools



def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)

    if agent_outcome:
        print(state)
        return None

    return {'agent_outcome': agent_outcome}



def act_node(state: AgentState):

    agent_action = state['agent_outcome']

    # extracting tool name and tool output 
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    # Find the matching tool function
    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    # Execute the tool with the input
    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool '{tool_name}' not found"
    
    return {"intermediate_steps": [(agent_action, str(output))]}


