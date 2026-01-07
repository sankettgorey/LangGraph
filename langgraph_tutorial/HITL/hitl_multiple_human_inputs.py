from typing import TypedDict, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# -------------------
# STATE SCHEMA
# -------------------
class HumanInTheLoopState(TypedDict):
    question: str
    ai_draft: str
    human_feedback: str
    final_response: str

# -------------------
# NODE DEFINITIONS
# -------------------

def draft_response(state: HumanInTheLoopState) -> Dict[str, Any]:
    draft = f"DRAFT: Here's my initial answer to: '{state['question']}'"
    print(f"\n🤖 AI drafted: {draft}\n")
    return {"ai_draft": draft}

def get_human_feedback(state: HumanInTheLoopState) -> Dict[str, Any]:
    # Pause here and ask for human input
    feedback = interrupt({
        "message": "Please review the draft and reply with 'approve' or feedback to revise:",
        "draft": state["ai_draft"],
    })
    print(f"\n👤 Human provided feedback: {feedback}\n")
    return {"human_feedback": feedback}

def decide_next_step(state: HumanInTheLoopState) -> Literal["revise", "finalize"]:
    if state["human_feedback"].strip().lower() == "approve":
        return "finalize"
    return "revise"

def revise_response(state: HumanInTheLoopState) -> Dict[str, Any]:
    revised = (
        f"REVISED: Updated based on human feedback: '{state['human_feedback']}'"
    )
    print(f"\n🤖 AI revised: {revised}\n")
    return {"ai_draft": revised}

def finalize_response(state: HumanInTheLoopState) -> Dict[str, Any]:
    final = f"FINAL: {state['ai_draft']}"
    print(f"\n✅ Final response: {final}\n")
    return {"final_response": final}

# -------------------
# BUILD GRAPH
# -------------------

def create_hitl_graph():
    graph = StateGraph(HumanInTheLoopState)

    graph.add_node("draft", draft_response)
    graph.add_node("human_review", get_human_feedback)
    graph.add_node("revise", revise_response)
    graph.add_node("finalize", finalize_response)

    graph.add_edge("draft", "human_review")
    graph.add_conditional_edges(
        "human_review",
        decide_next_step,
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "human_review")
    graph.add_edge("finalize", END)
    graph.set_entry_point("draft")

    # Must use a checkpointer because interrupts depend on saved state
    return graph.compile(checkpointer=InMemorySaver())

# -------------------
# EXECUTION LOOP
# -------------------

if __name__ == "__main__":
    workflow = create_hitl_graph()

    initial_state = {
        "question": "What's the best way to implement HITL systems?",
        "ai_draft": "",
        "human_feedback": "",
        "final_response": ""
    }

    config = {"configurable": {"thread_id": "chat-1"}}

    # This controller loop runs until no interrupt occurs
    current = initial_state
    while True:
        result = workflow.invoke(current, config=config)

        # If the graph paused for human input:
        if "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            print("\n--- HUMAN REVIEW REQUIRED ---")
            print(interrupt_payload["message"])
            print("Draft:", interrupt_payload["draft"])
            
            decision = input("Your decision [approve/reject/edit]: ")

            # Resume the graph with human input
            current = Command(resume=decision)
            continue

        # No interrupt → graph completed
        final_state = result
        break

    print("\n=== FINAL OUTPUT ===")
    print("Question:", final_state["question"])
    # print("AI Draft:", final_state["ai_draft"])
    # print("Human Feedback:", final_state["human_feedback"])
    print("Final Response:", final_state["final_response"])
    print("\n=== WORKFLOW COMPLETED ===")
