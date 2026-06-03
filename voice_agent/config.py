import os

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
TOOL_TIMEOUT_MS = 80
FSM_STATE_TIMEOUT_MS = {
    "greeting": 5_000,
    "taking_order": 8_000,
    "confirming": 6_000,
    "upsell": 5_000,
    "closing": 4_000,
}

# Menu Registry
MENU = {
    "burger": {"price": 5.99, "stock": 50, "synonyms": ["burgers", "hamburger", "hamburgers", "cheeseburger", "cheeseburgers", "double cheeseburger", "double cheeseburgers"]},
    "fries": {"price": 2.49, "stock": 100, "synonyms": ["french fries", "fry", "fries"]},
    "soda": {"price": 1.99, "stock": 200, "synonyms": ["coke", "cokes", "diet coke", "diet cokes", "sprite", "sprites", "soda", "sodas", "drink", "drinks", "pop"]},
    "nuggets": {"price": 4.49, "stock": 40, "synonyms": ["chicken nuggets", "nuggets", "nugs"]},
    "shake": {"price": 2.99, "stock": 15, "synonyms": ["milkshake", "milkshakes", "shake", "shakes", "chocolate shake", "vanilla shake", "strawberry shake"]},
    "onion rings": {"price": 2.99, "stock": 25, "synonyms": ["rings", "onion rings"]},
    "apple pie": {"price": 1.79, "stock": 0, "synonyms": ["pie", "apple pie"]} # Out of stock for testing!
}

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
        "You are a friendly, concise automated drive-thru ordering agent at 'Antigravity Burgers'. "
        "Your first response must greet the customer enthusiastically and ask what they would like to order. "
        "Keep it under 15 words."
    ),
    "taking_order": (
        "You are taking the customer's order. Call tools like check_inventory and add_to_cart as needed. "
        "Always summarize what is currently in their cart, specify any out-of-stock items, and ask "
        "if they would like to add anything else. Be extremely concise. Keep responses under 25 words."
    ),
    "confirming": (
        "You are confirming the complete order. Summarize the items and quantities, state the total price, "
        "and ask the customer to confirm if this is correct. Keep it under 25 words."
    ),
    "upsell": (
        "The customer confirmed their order. Now, attempt to upsell them exactly ONE of our high-margin items "
        "(e.g., onion rings or a milkshake/shake). Keep it friendly and under 15 words."
    ),
    "closing": (
        "The order is finalized. Tell the customer their total order amount, thank them, and instruct them "
        "to drive up to the next window. Keep it under 15 words."
    )
}
