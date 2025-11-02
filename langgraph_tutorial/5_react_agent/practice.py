from agent_reason_runnable import react_agent_runnable, tools

from react_state import AgentState


def reason_node(state: AgentState):

    agent_outcome = react_agent_runnable.invoke(state)
    

    return {
        'agent_outcome': agent_outcome
    }



def act(state: AgentState):

    agent_action = state['agent_outcome']

    # print(agent_action)

    tool_name = agent_action.tool
    tool_input = agent_action.tool_input


    # now finding matching tool function from the list of tools
    tool_function = None

    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)

        else:
            output = tool_function.invoke(tool_input)

    else:
        output = f'Tool {tool_name} not found'
    
    return {
        'intermediate_steps': [(agent_action, str(output))]
    }