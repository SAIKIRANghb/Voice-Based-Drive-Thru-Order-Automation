import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from voice_agent.config import FSM_STATE_TIMEOUT_MS, MENU, PROMPTS, TOOL_TIMEOUT_MS, get_gemini_api_key
from voice_agent.llm.tools import add_to_cart, apply_promo, check_inventory, get_price
from voice_agent.llm.guardrails import HallucinationGuard

# Initialize Hallucination Guard
guardrail = HallucinationGuard()
TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4)
DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_LLM_TIMEOUT_SECONDS = 20.0
AGENT_TOOLS = [check_inventory, get_price, add_to_cart, apply_promo]
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
    api_key = get_gemini_api_key()
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_LLM_MODEL", DEFAULT_GEMINI_LLM_MODEL)
        request_timeout = float(os.getenv("GEMINI_LLM_TIMEOUT_SECONDS", str(DEFAULT_GEMINI_LLM_TIMEOUT_SECONDS)))
        logging.info("Calling Gemini LLM model '%s' for node '%s'.", model_name, node_name)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            request_timeout=request_timeout,
            retries=int(os.getenv("GEMINI_LLM_RETRIES", "1")),
        )
        
        # Inject context about the cart and system prompt
        system_text = (
            PROMPTS.get(node_name, PROMPTS["taking_order"])
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
        content = response.content
        if isinstance(content, list):
            text = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content).strip()
            if not text:
                raise RuntimeError(f"Gemini returned empty text for node '{node_name}'.")
            return text
        text = str(content or "").strip()
        if not text:
            raise RuntimeError(f"Gemini returned empty text for node '{node_name}'.")
        return text
    except Exception:
        logging.exception(
            "Failed to get Gemini response with model '%s' for node '%s'. "
            "Set GEMINI_LLM_MODEL to a valid Gemini API model code if this model is unavailable.",
            os.getenv("GEMINI_LLM_MODEL", DEFAULT_GEMINI_LLM_MODEL),
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
    api_key = get_gemini_api_key()
    state = state or {}
    updated_cart = cart.copy()
    discount = float(state.get("discount", 0.0))
    free_items = list(state.get("free_items", []))

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = os.getenv("GEMINI_LLM_MODEL", DEFAULT_GEMINI_LLM_MODEL)
        request_timeout = float(os.getenv("GEMINI_LLM_TIMEOUT_SECONDS", str(DEFAULT_GEMINI_LLM_TIMEOUT_SECONDS)))
        logging.info("Calling Gemini tool agent model '%s' for node '%s'.", model_name, node_name)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            request_timeout=request_timeout,
            retries=int(os.getenv("GEMINI_LLM_RETRIES", "1")),
        ).bind_tools(AGENT_TOOLS)

        system_text = (
            PROMPTS.get(node_name, PROMPTS["taking_order"])
            + " Use tools for menu availability, pricing, cart changes, and promo codes. "
            + "If the customer asks for an item, calls a promo code, or accepts an upsell, call the matching tool before replying. "
            + "Do not invent unavailable items or prices. "
            + f"Menu: {menu_context()} "
            + order_context(updated_cart, discount, free_items)
        )
        formatted_messages = [SystemMessage(content=system_text), *messages[-5:]]
        if len(formatted_messages) == 1:
            formatted_messages.append(HumanMessage(content="Start the drive-thru conversation."))

        for _ in range(4):
            response = llm.invoke(formatted_messages)
            formatted_messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                content = response.content
                if isinstance(content, list):
                    text = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content).strip()
                else:
                    text = str(content or "").strip()
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
            os.getenv("GEMINI_LLM_MODEL", DEFAULT_GEMINI_LLM_MODEL),
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
    
    response_text = get_llm_response("upsell", messages, cart, state)
    validated_text, intercepted = guardrail.validate_response(response_text)
    
    discount = state.get("discount", 0.0)
    total_price = calculate_total(cart, discount)
    
    new_messages = messages + [AIMessage(content=validated_text)]
    
    return _state_update(
        state,
        messages=new_messages,
        cart=cart,
        total_price=total_price,
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
