import re
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from voice_agent.config import FSM_STATE_TIMEOUT_MS
from voice_agent.llm.nodes import (
    greeting_node, 
    taking_order_node, 
    confirming_node, 
    upsell_node, 
    closing_node
)

# Define the State structure
class AgentState(TypedDict):
    messages: List[BaseMessage]
    cart: Dict[str, int]
    total_price: float
    current_node: str
    next_node: str
    discount: float
    free_items: List[str]
    last_response: str
    hallucination_warning: bool
    state_timeout_ms: int

def get_last_user_message(state: AgentState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.lower()
    return ""

def route_turn(state: AgentState) -> str:
    """Select exactly one LangGraph FSM node to execute for this turn."""
    next_node = state.get("next_node") or state.get("current_node", "greeting")
    last_user_message = get_last_user_message(state)

    if not state.get("messages") and not last_user_message:
        return "greeting"

    done_keywords = [
        "that's all", "nothing else", "that is all", "done", "no more",
        "confirm order", "no thank you", "thats it", "ready"
    ]
    yes_keywords = ["yes", "yeah", "correct", "yep", "right", "confirm", "sure"]

    def has_trigger(text: str, triggers: list[str]) -> bool:
        return any(re.search(rf"\b{re.escape(trigger)}\b", text) for trigger in triggers)

    if next_node == "taking_order" and has_trigger(last_user_message, done_keywords):
        return "confirming"
    if next_node == "confirming":
        return "upsell" if has_trigger(last_user_message, yes_keywords) else "taking_order"
    if next_node == "upsell":
        return "closing"
    if next_node == "closing":
        return "closing"

    return next_node if next_node in {"greeting", "taking_order", "confirming", "upsell", "closing"} else "taking_order"

# Build the Graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("taking_order", taking_order_node)
    workflow.add_node("confirming", confirming_node)
    workflow.add_node("upsell", upsell_node)
    workflow.add_node("closing", closing_node)
    
    workflow.add_conditional_edges(
        START,
        route_turn,
        {
            "greeting": "greeting",
            "taking_order": "taking_order",
            "confirming": "confirming",
            "upsell": "upsell",
            "closing": "closing",
        },
    )
    workflow.add_edge("greeting", END)
    workflow.add_edge("taking_order", END)
    workflow.add_edge("confirming", END)
    workflow.add_edge("upsell", END)
    workflow.add_edge("closing", END)
    
    return workflow.compile()

# Compile standard instance
app_graph = build_graph()

def run_agent_turn(user_input: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single turn of the LangGraph agent.
    
    Args:
        user_input: The customer's transcribed voice text or simulated text.
        current_state: The existing state dict. If empty, starts a new conversation.
    
    Returns:
        The updated state dict.
    """
    # Initialize state if empty
    state = {
        "messages": list(current_state.get("messages", [])),
        "cart": current_state.get("cart", {}).copy(),
        "total_price": current_state.get("total_price", 0.0),
        "current_node": current_state.get("current_node", "greeting"),
        "next_node": current_state.get("next_node", current_state.get("current_node", "greeting")),
        "discount": current_state.get("discount", 0.0),
        "free_items": current_state.get("free_items", []),
        "last_response": current_state.get("last_response", ""),
        "hallucination_warning": current_state.get("hallucination_warning", False),
        "state_timeout_ms": current_state.get(
            "state_timeout_ms",
            FSM_STATE_TIMEOUT_MS.get(current_state.get("current_node", "greeting"), 5_000),
        ),
    }
    
    if not state["messages"] and not user_input:
        return app_graph.invoke(state)
        
    # Append the user's input message
    state["messages"].append(HumanMessage(content=user_input))
    
    return app_graph.invoke(state)
