import json
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from voice_agent.config import FSM_STATE_TIMEOUT_MS
from voice_agent.llm.nodes import (
    greeting_node, 
    taking_order_node, 
    confirming_node, 
    upsell_node, 
    closing_node
)
from voice_agent.llm.providers import get_llm_provider, response_json

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
            return str(msg.content)
    return ""

TRANSITION_MAP = {
    "greeting": [
        ("taking_order", "the customer starts ordering, asks menu questions, or responds to the greeting"),
    ],
    "taking_order": [
        ("taking_order", "the customer is adding items, browsing, asking menu questions, changing quantities, or the order is still incomplete"),
        ("confirming", "the customer says they are done, ready to checkout, ready to pay, or wants to review/confirm the order"),
    ],
    "confirming": [
        ("upsell", "the customer approves the reviewed order"),
        ("taking_order", "the customer changes, rejects, corrects, adds to, or asks a question about the reviewed order"),
    ],
    "upsell": [
        ("closing", "the customer accepts or declines the upsell and is ready to finish"),
        ("taking_order", "the customer changes the existing order or asks to keep ordering"),
    ],
    "closing": [
        ("taking_order", "the customer asks to modify the order"),
        ("closing", "the customer is done, says thanks, or the conversation should remain closed"),
    ],
}
ROUTABLE_NODES = set(TRANSITION_MAP)
GRAPH_NODE_MAP = {node: node for node in ROUTABLE_NODES}

def build_semantic_router_prompt() -> str:
    allowed_nodes = "\n".join(f"- {node}" for node in TRANSITION_MAP)
    transition_lines = "\n".join(
        f"- {source} -> {target} if {premise}."
        for source, transitions in TRANSITION_MAP.items()
        for target, premise in transitions
    )
    return f"""Choose the next LangGraph node for this drive-thru ordering FSM.
Return only compact JSON: {{"next_node":"..."}}.

Allowed nodes:
{allowed_nodes}

Transitions:
{transition_lines}

Route by meaning, not keywords. Use current_node, expected_next_node, cart, latest agent response, and latest customer utterance.
"""

SEMANTIC_ROUTER_PROMPT = build_semantic_router_prompt()

def semantic_next_node(state: AgentState, user_message: str) -> str:
    """Ask the LLM to choose the next graph node from explicit FSM transitions."""
    # A rule-based router can also be implemented here if deterministic routing is preferred.
    current_node = state.get("current_node", "greeting")
    expected_next_node = state.get("next_node", current_node)
    llm = get_llm_provider().chat(
        component="semantic router",
        model_env="GEMINI_ROUTER_MODEL",
        fallback_model_env="GEMINI_LLM_MODEL",
        timeout_env="GEMINI_ROUTER_TIMEOUT_SECONDS",
        default_timeout=10.0,
        retries_env="GEMINI_ROUTER_RETRIES",
        default_retries=0,
    )
    context = {
        "current_node": current_node,
        "expected_next_node": expected_next_node,
        "cart": state.get("cart", {}),
        "last_agent_response": state.get("last_response", ""),
        "latest_customer_utterance": user_message,
    }
    response = llm.invoke(
        [
            SystemMessage(content=SEMANTIC_ROUTER_PROMPT),
            HumanMessage(content=json.dumps(context, ensure_ascii=True)),
        ]
    )
    payload = response_json(response)
    next_node = str(payload.get("next_node", "")).strip().lower()
    if next_node not in ROUTABLE_NODES:
        raise ValueError(f"LLM router returned invalid next_node: {next_node!r}")
    return next_node

def route_turn(state: AgentState) -> str:
    """Select exactly one LangGraph FSM node to execute for this turn."""
    next_node = state.get("next_node") or state.get("current_node", "greeting")
    last_user_message = get_last_user_message(state)

    if not state.get("messages") and not last_user_message:
        return "greeting"

    return semantic_next_node(state, last_user_message)

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
        GRAPH_NODE_MAP,
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
