from langchain_core.tools import tool
from voice_agent.llm.tool_services import get_menu_tool_service

def resolve_menu_item(item_name: str) -> str:
    """Helper to match synonyms to official menu item keys."""
    return get_menu_tool_service().resolve_menu_item(item_name)

@tool
def list_products(page: int = 1, page_size: int = 3, category: str = "", query: str = "", top: bool = False) -> dict:
    """
    Return a paginated list of available products.
    Use this when the customer asks what is available, wants recommendations, asks for a category,
    or needs the top/bestselling products.
    """
    return get_menu_tool_service().list_products(page, page_size, category, query, top)


@tool
def get_product_details(item: str) -> dict:
    """
    Return full details for one product, including price, availability, description, tags, and add-on suggestions.
    Use this before answering detailed product questions or before asking customization follow-ups.
    """
    return get_menu_tool_service().get_product_details(item)


@tool
def suggest_addons(item: str = "", cart_items: str = "") -> dict:
    """
    Suggest add-ons, sides, drinks, or combo pairings for an item or current cart.
    Use this after adding an item, during confirmation if the cart is missing a drink/side,
    or during the upsell state.
    """
    return get_menu_tool_service().suggest_addons(item, cart_items)


@tool
def check_inventory(item_id: str) -> bool:
    """
    Check if a menu item is currently available in stock.
    Returns True if available, False otherwise.
    """
    return get_menu_tool_service().check_inventory(item_id)

@tool
def get_price(item: str) -> float:
    """
    Fetch the price of a specific menu item.
    Returns the float price of the item (e.g. 5.99). If not found, returns 0.0.
    """
    return get_menu_tool_service().get_price(item)

@tool
def add_to_cart(item: str, qty: int = 1) -> dict:
    """
    Add a quantity of an item to the customer's cart.
    Returns a dictionary summarizing the success and current details of the item.
    """
    return get_menu_tool_service().add_to_cart(item, qty)

@tool
def apply_promo(code: str) -> dict:
    """
    Validate and apply a promotional coupon code to the order.
    Supported codes: 'DISCOUNT10' (10% off), 'FREEFRIES' (free fries).
    """
    return get_menu_tool_service().apply_promo(code)

@tool
def search_menu_knowledge(query: str) -> str:
    """
    Retrieve relevant menu, inventory, category, and promo facts from Qdrant RAG.
    Use this before answering fuzzy menu questions or item recommendations.
    """
    return get_menu_tool_service().search_menu_knowledge(query, limit=4)
