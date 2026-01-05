from typing import TypedDict, Annotated, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


import json

# Define the state schema
class HumanInTheLoopState(TypedDict):
    question: str  # User's original question
    ai_draft: str  # AI's initial response draft
    human_feedback: str  # Human reviewer's feedback
    final_response: str  # Final response after incorporating feedback

# Create nodes for our workflow
def draft_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """AI drafts an initial response to the user's question"""
    question = state["question"]
    # In a real application, this would use an LLM to generate a response
    draft = f"DRAFT: Here's my initial answer to: '{question}'"
    
    print(f"\n🤖 AI has drafted a response: {draft}\n")
    return {"ai_draft": draft}

def get_human_feedback(state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect feedback from a human reviewer"""
    # In a real application, this would be implemented with a UI
    # or messaging platform to collect human input

    human_feedback = interrupt({
        "type": "Approve or Reject",
        "Question": state["question"],
        "initial_draft": state["ai_draft"],
        "decision": "[Approve/Reject]"
    })

    # print("\n👋 HUMAN REVIEW REQUIRED!\n")
    # print(f"Original question: {state['question']}")
    # print(f"AI draft: {state['ai_draft']}")
    
    # Simulating human input via console
    # feedback = input("\nPlease provide feedback or type 'approve' to accept: ")
    
    print(f"\n👤 Human provided feedback: {human_feedback}\n")
    return {"human_feedback": human_feedback}

def decide_next_step(state: Dict[str, Any]) -> Literal["revise", "finalize"]:
    """Decide whether to revise the response or finalize it"""
    print('in decision step')
    if state["human_feedback"].lower() == "approve":
        return "finalize"
    else:
        return "revise"

def revise_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Revise the response based on human feedback"""
    # In a real application, this would use an LLM to incorporate feedback
    print('in revise node')
    revised_response = f"REVISED: I've updated my answer based on feedback: '{state['human_feedback']}'"
    
    print(f"\n🤖 AI has revised the response: {revised_response}\n")
    return {"ai_draft": revised_response}

def finalize_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize the response and return to the user"""
    print('in finalize response step')
    final = f"FINAL: {state['ai_draft']}"
    
    print(f"\n✅ Final response ready: {final}\n")
    return {"final_response": final}


checkpointer = InMemorySaver()


# Create the graph
def create_hitl_graph():
    # Initialize the graph
    graph = StateGraph(HumanInTheLoopState)
    
    # Add nodes
    graph.add_node("draft", draft_response)
    graph.add_node("human_review", get_human_feedback)  # Renamed from "human_feedback"
    graph.add_node("revise", revise_response)
    graph.add_node("finalize", finalize_response)
    
    # Define the flow
    graph.add_edge("draft", "human_review")
    graph.add_conditional_edges(
        "human_review",
        decide_next_step,
        {
            "revise": "revise",
            "finalize": "finalize"
        }
    )
    graph.add_edge("revise", "human_review")
    graph.add_edge("finalize", END)
    
    # Set the entry point
    graph.set_entry_point("draft")
    
    return graph.compile(checkpointer=checkpointer)

# Usage example
if __name__ == "__main__":
    # Create the workflow
    workflow = create_hitl_graph()
    
    # Initial state with a user question
    initial_state = {
        "question": "What's the best way to implement HITL systems?",
        "ai_draft": "",
        "human_feedback": "",
        "final_response": ""
    }

    config={"configurable": {"thread_id": "1"}}
    
    # Run the workflow
    output = workflow.invoke(initial_state, config=config)

    human_feedback = output.get("__interrupt__", [])

    if human_feedback:
        prompt_for_user = human_feedback

        print(f"HITL: {prompt_for_user}")

        decision = input("Enter your decision [Approve/Reject]: ")

        output = workflow.invoke(Command(resume=decision),
                                 config=config)


    print(output['question'])
    print(output['ai_draft'])
    print(output['final_response'])
    print()
    # Final state contains the complete conversation
    print("\n=== WORKFLOW COMPLETED ===")