import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from voice_agent.data.interfaces import CatalogRepository

DATA_DIR = Path(__file__).resolve().parent
SEED_DATA_PATH = DATA_DIR / "swiggy_seed.json"


class JsonCatalogRepository(CatalogRepository):
    """JSON-backed catalog repository."""

    def __init__(self, seed_path: Path = SEED_DATA_PATH) -> None:
        self.seed_path = seed_path

    @lru_cache(maxsize=1)
    def load_catalog(self) -> dict[str, Any]:
        with self.seed_path.open("r", encoding="utf-8") as seed_file:
            return json.load(seed_file)


class SwiggyCatalogService:
    """Domain service for Swiggy-like restaurant catalog behavior."""

    def __init__(self, repository: CatalogRepository) -> None:
        self.repository = repository

    def load_catalog(self) -> dict[str, Any]:
        return self.repository.load_catalog()

    def menu_items(self) -> list[dict[str, Any]]:
        return list(self.load_catalog().get("menu_items", []))

    def offers(self) -> list[dict[str, Any]]:
        return list(self.load_catalog().get("offers", []))

    def categories(self) -> list[dict[str, Any]]:
        return list(self.load_catalog().get("categories", []))

    def category_name(self, category_id: str) -> str:
        for category in self.categories():
            if category["id"] == category_id:
                return category["name"]
        return "Menu"

    def available_menu_items(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.menu_items()
            if item.get("is_available", item.get("stock", 0) > 0) and item.get("stock", 0) > 0
        ]

    def top_menu_items(self, limit: int = 3) -> list[dict[str, Any]]:
        available = self.available_menu_items()
        return sorted(
            available,
            key=lambda item: (
                "bestseller" not in item.get("tags", []),
                "upsell" not in item.get("tags", []),
                item.get("category_id", ""),
                item.get("name", ""),
            ),
        )[:limit]

    def addon_suggestions_for_item(self, item_id: str) -> list[dict[str, Any]]:
        item_id = item_id.lower().strip()
        if item_id in {"burger", "nuggets"}:
            return [
                {"item": "fries", "reason": "classic side combo"},
                {"item": "soda", "reason": "drink pairing"},
                {"item": "shake", "reason": "premium drink upgrade"},
            ]
        if item_id in {"fries", "onion rings"}:
            return [
                {"item": "soda", "reason": "drink pairing"},
                {"item": "burger", "reason": "main item"},
            ]
        if item_id == "soda":
            return [
                {"item": "fries", "reason": "quick snack pairing"},
                {"item": "burger", "reason": "main item"},
            ]
        if item_id == "shake":
            return [
                {"item": "burger", "reason": "main item"},
                {"item": "fries", "reason": "salty side pairing"},
            ]
        return [{"item": "fries", "reason": "popular add-on"}]

    def legacy_menu(self) -> dict[str, dict[str, Any]]:
        menu: dict[str, dict[str, Any]] = {}
        for item in self.menu_items():
            item_id = item["id"]
            menu[item_id] = {
                "price": float(item["price"]),
                "stock": int(item.get("stock", 0)),
                "synonyms": list(item.get("synonyms", [])),
                "description": item.get("description", ""),
                "category_id": item.get("category_id", ""),
                "tags": list(item.get("tags", [])),
                "is_available": bool(item.get("is_available", item.get("stock", 0) > 0)),
            }
        return menu

    def catalog_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        categories = {category["id"]: category["name"] for category in self.categories()}
        for item in self.menu_items():
            available_text = "available" if item.get("is_available") and item.get("stock", 0) > 0 else "out of stock"
            text = (
                f"{item['name']} costs ${float(item['price']):.2f}. "
                f"Category: {categories.get(item.get('category_id'), 'Menu')}. "
                f"{item.get('description', '')} "
                f"Availability: {available_text}. "
                f"Synonyms: {', '.join(item.get('synonyms', []))}. "
                f"Tags: {', '.join(item.get('tags', []))}."
            ).strip()
            documents.append(
                {
                    "id": item["id"],
                    "kind": "menu_item",
                    "text": text,
                    "metadata": {
                        "item_id": item["id"],
                        "name": item["name"],
                        "price": float(item["price"]),
                        "stock": int(item.get("stock", 0)),
                        "is_available": bool(item.get("is_available", item.get("stock", 0) > 0)),
                        "category": categories.get(item.get("category_id"), "Menu"),
                        "tags": list(item.get("tags", [])),
                        "synonyms": list(item.get("synonyms", [])),
                    },
                }
            )

        for offer in self.offers():
            text = f"Promo code {offer['code']}: {offer['title']}."
            documents.append({"id": offer["id"], "kind": "offer", "text": text, "metadata": offer})
        return documents


@lru_cache(maxsize=1)
def get_default_catalog_service() -> SwiggyCatalogService:
    return SwiggyCatalogService(JsonCatalogRepository())


def load_catalog() -> dict[str, Any]:
    return get_default_catalog_service().load_catalog()


def menu_items() -> list[dict[str, Any]]:
    return get_default_catalog_service().menu_items()


def offers() -> list[dict[str, Any]]:
    return get_default_catalog_service().offers()


def categories() -> list[dict[str, Any]]:
    return get_default_catalog_service().categories()


def category_name(category_id: str) -> str:
    return get_default_catalog_service().category_name(category_id)


def available_menu_items() -> list[dict[str, Any]]:
    return get_default_catalog_service().available_menu_items()


def top_menu_items(limit: int = 3) -> list[dict[str, Any]]:
    return get_default_catalog_service().top_menu_items(limit)


def addon_suggestions_for_item(item_id: str) -> list[dict[str, Any]]:
    return get_default_catalog_service().addon_suggestions_for_item(item_id)


def legacy_menu() -> dict[str, dict[str, Any]]:
    return get_default_catalog_service().legacy_menu()


def catalog_documents() -> list[dict[str, Any]]:
    return get_default_catalog_service().catalog_documents()
