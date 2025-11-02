## Visual Representation
```
┌─────────────────────────────────────────────────────────┐
│ User Input: "What's 25 * 4?"                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Agent Node (Reason/Thought)                             │
│ - LLM receives: [HumanMessage]                          │
│ - Decides: Use calculator tool                          │
│ - Returns: AIMessage with tool_calls (AgentAction)      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Router (should_continue)                                │
│ - Checks: Does message have tool_calls?                 │
│ - Decision: "continue" → go to tools                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Tool Node (Action)                                      │
│ - LangGraph runtime executes: calculator("25 * 4")      │
│ - Result: "100"                                         │
│ - Adds: ToolMessage to state (Observation)              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ State Update (Scratchpad/Intermediate Steps)            │
│ messages = [                                            │
│   HumanMessage("What's 25 * 4?"),                       │
│   AIMessage(tool_calls=[calculator]),                   │
│   ToolMessage("100")  ← Observation                     │
│ ]                                                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Back to Agent Node (with updated scratchpad)            │
│ - LLM receives: All messages including tool result      │
│ - Sees observation: "100"                               │
│ - Decides: I have the answer                            │
│ - Returns: AIMessage with NO tool_calls (AgentFinish)   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Router (should_continue)                                │
│ - Checks: No tool_calls found                           │
│ - Decision: "end" → finish                              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Final Output Parsing                                    │
│ - Extract: result["messages"][-1].content               │
│ - User receives: "25 * 4 = 100"                         │
└─────────────────────────────────────────────────────────┘