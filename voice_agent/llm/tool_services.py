from functools import lru_cache
from typing import Any

from voice_agent.data.factories import get_catalog_service, get_menu_retriever
from voice_agent.data.interfaces import CatalogService, MenuRetriever


class MenuToolService:
    """Application service behind LLM menu/order tools."""

    def __init__(self, catalog_service: CatalogService, retriever: MenuRetriever) -> None:
        self.catalog_service = catalog_service
        self.retriever = retriever

    @property
    def menu(self) -> dict[str, dict[str, Any]]:
        return self.catalog_service.legacy_menu()

    def resolve_menu_item(self, item_name: str) -> str:
        name = item_name.lower().strip()
        for official_name, details in self.menu.items():
            if name == official_name or name in details.get("synonyms", []):
                return official_name
        return name

    def list_products(
        self,
        page: int = 1,
        page_size: int = 3,
        category: str = "",
        query: str = "",
        top: bool = False,
    ) -> dict:
        page = max(1, int(page or 1))
        page_size = min(5, max(1, int(page_size or 3)))
        normalized_category = category.lower().strip()
        normalized_query = query.lower().strip()

        items = self.catalog_service.top_menu_items(limit=page_size) if top else self.catalog_service.available_menu_items()
        if normalized_category:
            items = [
                item
                for item in items
                if normalized_category in self.catalog_service.category_name(item.get("category_id", "")).lower()
                or normalized_category in item.get("category_id", "").lower()
            ]
        if normalized_query:
            items = [
                item
                for item in items
                if normalized_query in item["name"].lower()
                or normalized_query in item.get("description", "").lower()
                or any(normalized_query in synonym.lower() for synonym in item.get("synonyms", []))
                or any(normalized_query in tag.lower() for tag in item.get("tags", []))
            ]

        total = len(items)
        start = (page - 1) * page_size
        page_items = items[start : start + page_size]
        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": start + page_size < total,
            "products": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "price": float(item["price"]),
                    "category": self.catalog_service.category_name(item.get("category_id", "")),
                    "description": item.get("description", ""),
                    "tags": item.get("tags", []),
                }
                for item in page_items
            ],
        }

    def get_product_details(self, item: str) -> dict:
        official_name = self.resolve_menu_item(item)
        if official_name not in self.menu:
            return {"success": False, "error": f"Item '{item}' is not on the menu."}

        details = self.menu[official_name]
        return {
            "success": True,
            "id": official_name,
            "price": float(details["price"]),
            "stock": int(details["stock"]),
            "is_available": details["stock"] > 0 and details.get("is_available", True),
            "description": details.get("description", ""),
            "category_id": details.get("category_id", ""),
            "tags": details.get("tags", []),
            "addon_suggestions": self.catalog_service.addon_suggestions_for_item(official_name),
        }

    def suggest_addons(self, item: str = "", cart_items: str = "") -> dict:
        candidates: list[dict] = []
        if item:
            candidates.extend(self.catalog_service.addon_suggestions_for_item(self.resolve_menu_item(item)))
        for cart_item in [part.strip() for part in cart_items.split(",") if part.strip()]:
            candidates.extend(self.catalog_service.addon_suggestions_for_item(self.resolve_menu_item(cart_item)))

        seen = set()
        products = []
        for candidate in candidates or self.catalog_service.addon_suggestions_for_item("burger"):
            candidate_item = self.resolve_menu_item(candidate["item"])
            if candidate_item in seen or candidate_item not in self.menu or self.menu[candidate_item]["stock"] <= 0:
                continue
            seen.add(candidate_item)
            products.append(
                {
                    "id": candidate_item,
                    "price": float(self.menu[candidate_item]["price"]),
                    "reason": candidate["reason"],
                }
            )
            if len(products) >= 3:
                break

        return {"suggestions": products}

    def check_inventory(self, item_id: str) -> bool:
        official_name = self.resolve_menu_item(item_id)
        if official_name in self.menu:
            return self.menu[official_name]["stock"] > 0
        return False

    def get_price(self, item: str) -> float:
        official_name = self.resolve_menu_item(item)
        if official_name in self.menu:
            return self.menu[official_name]["price"]
        return 0.0

    def add_to_cart(self, item: str, qty: int = 1) -> dict:
        official_name = self.resolve_menu_item(item)
        qty = max(1, int(qty or 1))
        if official_name not in self.menu:
            return {"success": False, "error": f"Item '{item}' is not on the menu."}

        stock = self.menu[official_name]["stock"]
        if stock <= 0:
            return {"success": False, "error": f"Item '{official_name}' is currently out of stock."}

        price = self.menu[official_name]["price"]
        qty_added = min(qty, stock)
        return {
            "success": True,
            "item": official_name,
            "quantity": qty_added,
            "price": price,
            "total": round(price * qty_added, 2),
        }

    def apply_promo(self, code: str) -> dict:
        code_upper = code.upper().strip()
        for offer in self.catalog_service.offers():
            if offer.get("code", "").upper() != code_upper or not offer.get("is_active", True):
                continue
            if offer.get("discount_type") == "percent":
                return {
                    "success": True,
                    "discount_type": "percent",
                    "value": float(offer.get("value", 0.0)),
                    "message": f"{offer['title']} coupon applied!",
                }
            if offer.get("discount_type") == "free_item":
                return {
                    "success": True,
                    "discount_type": "free_item",
                    "item": offer.get("item_id"),
                    "message": f"{offer['title']} coupon applied!",
                }
        return {"success": False, "error": "Invalid promo code."}

    def search_menu_knowledge(self, query: str, limit: int = 4) -> str:
        matches = self.retriever.search(query, limit)
        return " ".join(match["text"] for match in matches) or "No matching menu knowledge found."


@lru_cache(maxsize=1)
def get_menu_tool_service() -> MenuToolService:
    return MenuToolService(get_catalog_service(), get_menu_retriever())
