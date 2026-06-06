from typing import Any, Protocol


class CatalogRepository(Protocol):
    def load_catalog(self) -> dict[str, Any]:
        """Return the raw restaurant catalog."""


class CatalogService(Protocol):
    def menu_items(self) -> list[dict[str, Any]]:
        """Return all menu items."""

    def offers(self) -> list[dict[str, Any]]:
        """Return configured offers."""

    def categories(self) -> list[dict[str, Any]]:
        """Return menu categories."""

    def available_menu_items(self) -> list[dict[str, Any]]:
        """Return currently available menu items."""

    def top_menu_items(self, limit: int = 3) -> list[dict[str, Any]]:
        """Return top products for discovery."""

    def category_name(self, category_id: str) -> str:
        """Resolve a category id to a display name."""

    def addon_suggestions_for_item(self, item_id: str) -> list[dict[str, Any]]:
        """Return add-on suggestions for a product."""

    def legacy_menu(self) -> dict[str, dict[str, Any]]:
        """Expose the old MENU shape used by existing order code."""

    def catalog_documents(self) -> list[dict[str, Any]]:
        """Return retrieval-ready catalog documents."""


class MenuRetriever(Protocol):
    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        """Search menu knowledge."""


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return a dense vector for text."""
