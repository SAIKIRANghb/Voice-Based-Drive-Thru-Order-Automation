from langchain_core.tools import tool
from voice_agent.config import MENU

def resolve_menu_item(item_name: str) -> str:
    """Helper to match synonyms to official menu item keys."""
    name = item_name.lower().strip()
    for official_name, details in MENU.items():
        if name == official_name or name in details.get("synonyms", []):
            return official_name
    return name

@tool
def check_inventory(item_id: str) -> bool:
    """
    Check if a menu item is currently available in stock.
    Returns True if available, False otherwise.
    """
    official_name = resolve_menu_item(item_id)
    if official_name in MENU:
        return MENU[official_name]["stock"] > 0
    return False

@tool
def get_price(item: str) -> float:
    """
    Fetch the price of a specific menu item.
    Returns the float price of the item (e.g. 5.99). If not found, returns 0.0.
    """
    official_name = resolve_menu_item(item)
    if official_name in MENU:
        return MENU[official_name]["price"]
    return 0.0

@tool
def add_to_cart(item: str, qty: int = 1) -> dict:
    """
    Add a quantity of an item to the customer's cart.
    Returns a dictionary summarizing the success and current details of the item.
    """
    official_name = resolve_menu_item(item)
    qty = max(1, int(qty or 1))
    if official_name not in MENU:
        return {"success": False, "error": f"Item '{item}' is not on the menu."}
        
    stock = MENU[official_name]["stock"]
    if stock <= 0:
        return {"success": False, "error": f"Item '{official_name}' is currently out of stock."}
        
    price = MENU[official_name]["price"]
    qty_added = min(qty, stock)
    
    return {
        "success": True, 
        "item": official_name, 
        "quantity": qty_added, 
        "price": price,
        "total": round(price * qty_added, 2)
    }

@tool
def apply_promo(code: str) -> dict:
    """
    Validate and apply a promotional coupon code to the order.
    Supported codes: 'DISCOUNT10' (10% off), 'FREEFRIES' (free fries).
    """
    code_upper = code.upper().strip()
    if code_upper == "DISCOUNT10":
        return {"success": True, "discount_type": "percent", "value": 0.10, "message": "10% off coupon applied!"}
    elif code_upper == "FREEFRIES":
        return {"success": True, "discount_type": "free_item", "item": "fries", "message": "Free Fries coupon applied!"}
    else:
        return {"success": False, "error": "Invalid promo code."}
