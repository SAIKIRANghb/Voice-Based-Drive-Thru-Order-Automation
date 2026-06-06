import os
from voice_agent.data.catalog import legacy_menu

# Audio / DSP settings from the LLD.
SAMPLE_RATE = 16_000
FRAME_SIZE = 320  # 20 ms at 16 kHz
REFERENCE_BUFFER_SECONDS = 2
REFERENCE_BUFFER_SAMPLES = SAMPLE_RATE * REFERENCE_BUFFER_SECONDS

def get_gemini_api_key() -> str:
    """Read Gemini API key from the preferred env var name."""
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if api_key:
        return api_key

    legacy_api_key = (os.getenv("GEMINI_KEY_API") or "").strip()
    if legacy_api_key:
        return legacy_api_key

    raise RuntimeError("GEMINI_API_KEY is required.")

NLMS_TAPS = 512
NLMS_MU = 0.05
NLMS_EPS = 1e-6
VAD_THRESHOLD = 0.60

# LLM/FSM settings from the LLD.
TOOL_TIMEOUT_MS = int(os.getenv("TOOL_TIMEOUT_MS", "1500"))
FSM_STATE_TIMEOUT_MS = {
    "greeting": 5_000,
    "taking_order": 8_000,
    "confirming": 6_000,
    "upsell": 5_000,
    "closing": 4_000,
}

# Menu Registry, loaded from the Swiggy-like catalog seed.
MENU = legacy_menu()

# Whitelist for Hallucination Guard
MENU_WHITELIST = set(MENU.keys()) | {
    "burgers", "hamburger", "hamburgers", "cheeseburger", "cheeseburgers",
    "double cheeseburger", "double cheeseburgers", "coke", "cokes", "sprite",
    "sprites", "diet coke", "diet cokes", "milkshake", "milkshakes", "drink",
    "drinks", "pop", "french fries", "fry",
    "chicken nuggets", "nugs", "onion rings", "rings", "apple pie", "pie",
    "fries", "nuggets", "shake", "soda",
}

# General food items used by the hallucination guard. Any detected food phrase
# outside MENU_WHITELIST is clarified before TTS synthesis.
FOOD_NOUNS = {
    "apple pie", "burger", "burrito", "cheeseburger", "chicken nuggets", "coke",
    "diet coke", "diet cokes", "drink", "drinks", "fries", "french fries",
    "hamburger", "hamburgers", "hot dog", "hotdog", "milkshake", "milkshakes",
    "nuggets", "onion rings", "pasta", "pizza", "pop", "salad", "sandwich",
    "shake", "shakes", "soda", "sodas", "sprite", "sprites", "sushi", "tacos",
}
NON_MENU_ITEMS = FOOD_NOUNS - MENU_WHITELIST

CLARIFY_TEMPLATE = (
    "I'm sorry, I didn't catch that. Did you mean {suggestions}? "
    "Our menu has burgers, nuggets, fries, onion rings, shakes, and soft drinks."
)

# Prompts for different states
PROMPTS = {
    "greeting": (
        "You are a friendly automated drive-thru ordering agent at 'Antigravity Burgers'. "
        "Start the buying workflow: greet the customer, mention that you can show top picks or categories, "
        "and ask what they would like. If relevant product context is provided, mention at most two top items. "
        "Keep it natural and under 22 words."
    ),
    "taking_order": (
        "You are in the browse/select/customize stage of a real food-ordering workflow. "
        "Use list_products(page, page_size, category, query, top) when the customer asks what is available, "
        "asks for top items, asks by category, or seems undecided; offer the next page only when has_next is true. "
        "Use search_menu_knowledge for fuzzy questions, recommendations, dietary/category/promo questions, or semantic matches. "
        "Use get_product_details before answering detailed item questions or asking customization follow-ups. "
        "Use check_inventory and add_to_cart when the customer names an item and quantity. "
        "After adding an item, summarize the cart briefly and ask one useful follow-up: size/flavor when relevant, "
        "or whether they want a side, drink, sauce, or combo add-on. "
        "If an item is unavailable, apologize, name one close available alternative, and ask whether to add it. "
        "Do not invent items, variants, prices, stock, or offers. Keep responses under 35 words."
    ),
    "confirming": (
        "You are in cart review before checkout. Summarize item quantities, free items, discounts, and total price. "
        "If the cart lacks a drink or side, ask one final add-on question using provided retrieved/add-on context; "
        "otherwise ask for confirmation. If the customer changes anything, route back to order taking. "
        "Do not add new items unless the customer explicitly agrees. Keep it under 35 words."
    ),
    "upsell": (
        "The customer confirmed the core order. Make exactly one relevant, low-pressure upsell using suggest_addons "
        "or retrieved menu context: prefer drinks, sides, shakes, or a promo-compatible item not already in the cart. "
        "If they accept, use add_to_cart before replying; if they decline, move to checkout. "
        "Keep it friendly and under 22 words."
    ),
    "closing": (
        "You are closing checkout. Confirm the final cart and total, mention applied promo/free item if any, "
        "thank the customer, and direct them to the next window for payment/pickup. "
        "If the customer accepts the previous upsell, call add_to_cart before finalizing. "
        "If they ask to modify the order, do not finalize; ask the needed follow-up. Keep it under 28 words."
    )
}
