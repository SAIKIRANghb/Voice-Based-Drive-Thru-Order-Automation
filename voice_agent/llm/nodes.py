import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from voice_agent.config import FSM_STATE_TIMEOUT_MS, MENU, PROMPTS, TOOL_TIMEOUT_MS
from voice_agent.data.catalog import addon_suggestions_for_item, top_menu_items
from voice_agent.data.qdrant_rag import retrieve_menu_context
from voice_agent.llm.providers import DEFAULT_GEMINI_LLM_MODEL, get_llm_provider, response_text
from voice_agent.llm.tools import (
    add_to_cart,
    apply_promo,
    check_inventory,
    get_price,
    get_product_details,
    list_products,
    search_menu_knowledge,
    suggest_addons,
)
from voice_agent.llm.guardrails import HallucinationGuard

# Initialize Hallucination Guard
guardrail = HallucinationGuard()
TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4)
AGENT_TOOLS = [
    list_products,
    get_product_details,
    suggest_addons,
    check_inventory,
    get_price,
    add_to_cart,
    apply_promo,
    search_menu_knowledge,
]
AGENT_TOOL_BY_NAME = {tool.name: tool for tool in AGENT_TOOLS}

def _state_update(state: Dict[str, Any], **updates) -> Dict[str, Any]:
    """Return a full LangGraph state so cross-node fields are not dropped."""
    next_state = {
        "messages": state.get("messages", []),
        "cart": state.get("cart", {}),
        "total_price": state.get("total_price", 0.0),
        "current_node": state.get("current_node", "greeting"),
        "next_node": state.get("next_node", state.get("current_node", "greeting")),
        "discount": state.get("discount", 0.0),
        "free_items": state.get("free_items", []),
        "last_response": state.get("last_response", ""),
        "hallucination_warning": state.get("hallucination_warning", False),
        "state_timeout_ms": state.get("state_timeout_ms", FSM_STATE_TIMEOUT_MS["greeting"]),
    }
    next_state.update(updates)
    return next_state

def invoke_tool_with_timeout(tool, args: dict):
    """Run an LLD tool call with the configured agent timeout."""
    future = TOOL_EXECUTOR.submit(tool.invoke, args)
    try:
        return future.result(timeout=TOOL_TIMEOUT_MS / 1000)
    except TimeoutError:
        logging.exception("Tool '%s' exceeded %sms timeout.", tool.name, TOOL_TIMEOUT_MS)
        raise
    except Exception as exc:
        logging.exception("Tool '%s' failed: %s", tool.name, exc)
        raise

def format_cart(cart: Dict[str, int]) -> str:
    if not cart:
        return "no items"
    return ", ".join(f"{qty} {item}" for item, qty in cart.items())

def calculate_total(cart: Dict[str, int], discount: float = 0.0) -> float:
    subtotal = sum(MENU[item]["price"] * qty for item, qty in cart.items())
    return round(subtotal * (1.0 - discount), 2)

def menu_context() -> str:
    items = []
    for item, details in MENU.items():
        stock_text = "available" if details["stock"] > 0 else "out of stock"
        items.append(f"{item}: ${details['price']:.2f}, {stock_text}")
    return "; ".join(items)

def top_products_context(limit: int = 3) -> str:
    items = top_menu_items(limit)
    if not items:
        return "none"
    return "; ".join(f"{item['name']}: ${float(item['price']):.2f}" for item in items)

def rag_context_for_messages(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return retrieve_menu_context(str(msg.content), limit=4)
    return ""

def addon_context(cart: Dict[str, int]) -> str:
    suggestions = []
    seen = set(cart.keys())
    for item in cart:
        for suggestion in addon_suggestions_for_item(item):
            suggestion_item = suggestion["item"]
            if suggestion_item in seen or suggestion_item not in MENU or MENU[suggestion_item]["stock"] <= 0:
                continue
            seen.add(suggestion_item)
            suggestions.append(f"{suggestion_item} (${MENU[suggestion_item]['price']:.2f}, {suggestion['reason']})")
            if len(suggestions) >= 3:
                return "; ".join(suggestions)
    return "; ".join(suggestions) or "fries, soda, or shake"

def order_context(cart: Dict[str, int], discount: float = 0.0, free_items: list[str] | None = None) -> str:
    return (
        f"Current cart: {format_cart(cart)}. "
        f"Total: ${calculate_total(cart, discount):.2f}. "
        f"Applied discount: {discount:.0%}. "
        f"Free items: {', '.join(free_items or []) or 'none'}."
    )

def apply_tool_result_to_order(
    tool_name: str,
    result,
    cart: Dict[str, int],
    discount: float,
    free_items: list[str],
) -> tuple[Dict[str, int], float, list[str]]:
    updated_cart = cart.copy()
    updated_free_items = list(free_items)
    updated_discount = discount

    if tool_name == "add_to_cart" and isinstance(result, dict) and result.get("success"):
        item = result["item"]
        updated_cart[item] = updated_cart.get(item, 0) + int(result.get("quantity", 1))
    elif tool_name == "apply_promo" and isinstance(result, dict) and result.get("success"):
        if result.get("discount_type") == "percent":
            updated_discount = float(result.get("value", discount))
        elif result.get("discount_type") == "free_item":
            item = result.get("item")
            if item and item in MENU and MENU[item]["stock"] > 0:
                updated_cart[item] = updated_cart.get(item, 0) + 1
                updated_free_items.append(item)

    return updated_cart, updated_discount, updated_free_items

def get_llm_response(node_name: str, messages: list, cart: dict, state: dict | None = None) -> str:
    """
    Get a response for a node from Gemini.
    """
    try:
        llm = get_llm_provider().chat(component=f"node '{node_name}'")
        
        # Inject context about the cart and system prompt
        system_text = (
            PROMPTS.get(node_name, PROMPTS["taking_order"])
            + " Workflow facts: "
            + f"Top products: {top_products_context()}. "
            + f"Relevant retrieved menu knowledge: {rag_context_for_messages(messages) or 'none'}. "
            + f"Suggested add-ons for current cart: {addon_context(cart)}. "
            + f" {order_context(cart, float((state or {}).get('discount', 0.0)), (state or {}).get('free_items', []))}"
        )
        
        # Build messages list
        formatted_messages = [SystemMessage(content=system_text)]
        for msg in messages[-5:]: # Look back at last 5 messages for speed
            formatted_messages.append(msg)
        if len(formatted_messages) == 1:
            formatted_messages.append(HumanMessage(content="Start the drive-thru conversation."))
            
        # Get response
        response = llm.invoke(formatted_messages)
        text = response_text(response)
        if not text:
            raise RuntimeError(f"Gemini returned empty text for node '{node_name}'.")
        return text
    except Exception:
        logging.exception(
            "Failed to get Gemini response with model '%s' for node '%s'. "
            "Set GEMINI_LLM_MODEL to a valid Gemini API model code if this model is unavailable.",
            DEFAULT_GEMINI_LLM_MODEL,
            node_name,
        )
        raise

def get_agent_response(
    node_name: str,
    messages: list,
    cart: dict,
    state: dict | None = None,
) -> tuple[str, Dict[str, int], float, list[str]]:
    """Run the Gemini agent with tools and return its final customer response."""
    state = state or {}
    updated_cart = cart.copy()
    discount = float(state.get("discount", 0.0))
    free_items = list(state.get("free_items", []))

    try:
        llm = get_llm_provider().chat(component=f"tool agent node '{node_name}'", tools=AGENT_TOOLS)

        system_text = (
            PROMPTS.get(node_name, PROMPTS["taking_order"])
            + " Use tools for paginated product browsing, RAG lookup, product details, add-on suggestions, menu availability, pricing, cart changes, and promo codes. "
            + "Use list_products for top products, category browsing, search results, or pagination. "
            + "Use search_menu_knowledge for fuzzy menu questions, dietary questions, recommendations, categories, and promo details. "
            + "Use get_product_details before specific product follow-ups, and suggest_addons before add-on or combo suggestions. "
            + "If the customer asks for an item, calls a promo code, or accepts an upsell, call the matching tool before replying. "
            + "Do not invent unavailable items or prices. "
            + f"Menu: {menu_context()} "
            + f"Top products: {top_products_context()}. "
            + f"Retrieved menu knowledge: {rag_context_for_messages(messages) or 'none'}. "
            + f"Suggested add-ons for current cart: {addon_context(updated_cart)}. "
            + order_context(updated_cart, discount, free_items)
        )
        formatted_messages = [SystemMessage(content=system_text), *messages[-5:]]
        if len(formatted_messages) == 1:
            formatted_messages.append(HumanMessage(content="Start the drive-thru conversation."))

        for _ in range(6):
            response = llm.invoke(formatted_messages)
            formatted_messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                text = response_text(response)
                if not text:
                    raise RuntimeError(f"Gemini returned empty text for node '{node_name}'.")
                return text, updated_cart, discount, free_items

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool = AGENT_TOOL_BY_NAME.get(tool_name)
                if tool is None:
                    result = {"success": False, "error": f"Unknown tool '{tool_name}'."}
                else:
                    result = invoke_tool_with_timeout(tool, tool_call.get("args", {}))
                    updated_cart, discount, free_items = apply_tool_result_to_order(
                        tool_name,
                        result,
                        updated_cart,
                        discount,
                        free_items,
                    )
                formatted_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )

        raise RuntimeError(f"Gemini did not produce a final response for node '{node_name}' after tool calls.")
    except Exception:
        logging.exception(
            "Failed to run Gemini tool agent with model '%s' for node '%s'. "
            "Set GEMINI_LLM_MODEL to a valid Gemini API model code if this model is unavailable.",
            DEFAULT_GEMINI_LLM_MODEL,
            node_name,
        )
        raise

# --- LANGGRAPH NODE IMPLEMENTATIONS ---

def greeting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Greeting state: welcomes the customer and prompts for order."""
    messages = state.get("messages", [])
    cart = state.get("cart", {})
    
    response_text = get_llm_response("greeting", messages, cart, state)
    
    # Intercept with Hallucination Guard
    validated_text, intercepted = guardrail.validate_response(response_text)
    
    new_messages = messages + [AIMessage(content=validated_text)]
    
    return _state_update(
        state,
        messages=new_messages,
        current_node="greeting",
        next_node="taking_order",
        last_response=validated_text,
        hallucination_warning=intercepted,
        state_timeout_ms=FSM_STATE_TIMEOUT_MS["greeting"],
    )

def taking_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Taking order state: lets the Gemini agent use tools to update the cart."""
    messages = state.get("messages", [])
    cart = state.get("cart", {}).copy()

    response_text, cart, discount, free_items = get_agent_response("taking_order", messages, cart, state)

    # Intercept with Hallucination Guard
    validated_text, intercepted = guardrail.validate_response(response_text)
    
    # Recalculate total price
    total_price = calculate_total(cart, discount)
    
    new_messages = messages + [AIMessage(content=validated_text)]
    
    return _state_update(
        state,
        messages=new_messages,
        cart=cart,
        total_price=total_price,
        discount=discount,
        free_items=free_items,
        current_node="taking_order",
        next_node="taking_order",
        last_response=validated_text,
        hallucination_warning=intercepted,
        state_timeout_ms=FSM_STATE_TIMEOUT_MS["taking_order"],
    )

def confirming_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Confirming state: asks Gemini to confirm the full order."""
    messages = state.get("messages", [])
    cart = state.get("cart", {})
    
    total_price = calculate_total(cart, state.get("discount", 0.0))
    response_text = get_llm_response("confirming", messages, cart, state)
    validated_text, intercepted = guardrail.validate_response(response_text)
    
    new_messages = messages + [AIMessage(content=validated_text)]
    
    return _state_update(
        state,
        messages=new_messages,
        current_node="confirming",
        next_node="confirming",
        last_response=validated_text,
        hallucination_warning=intercepted,
        state_timeout_ms=FSM_STATE_TIMEOUT_MS["confirming"],
    )

def upsell_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Upsell state: offers high-margin sides/shakes."""
    messages = state.get("messages", [])
    cart = state.get("cart", {}).copy()
    
    response_text, cart, discount, free_items = get_agent_response("upsell", messages, cart, state)
    validated_text, intercepted = guardrail.validate_response(response_text)
    
    total_price = calculate_total(cart, discount)
    
    new_messages = messages + [AIMessage(content=validated_text)]
    
    return _state_update(
        state,
        messages=new_messages,
        cart=cart,
        total_price=total_price,
        discount=discount,
        free_items=free_items,
        current_node="upsell",
        next_node="upsell",
        last_response=validated_text,
        hallucination_warning=intercepted,
        state_timeout_ms=FSM_STATE_TIMEOUT_MS["upsell"],
    )

def closing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Closing state: asks Gemini to close the conversation."""
    messages = state.get("messages", [])
    cart = state.get("cart", {}).copy()
    
    discount = state.get("discount", 0.0)
    free_items = state.get("free_items", [])
    response_text, cart, discount, free_items = get_agent_response("closing", messages, cart, state)
    total_price = calculate_total(cart, discount)
    validated_text, intercepted = guardrail.validate_response(response_text)
    
    new_messages = messages + [AIMessage(content=validated_text)]
    
    return _state_update(
        state,
        messages=new_messages,
        cart=cart,
        total_price=total_price,
        discount=discount,
        free_items=free_items,
        current_node="closing",
        next_node="closing",
        last_response=validated_text,
        hallucination_warning=intercepted,
        state_timeout_ms=FSM_STATE_TIMEOUT_MS["closing"],
    )
