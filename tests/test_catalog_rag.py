import unittest

from voice_agent.config import MENU
from voice_agent.data.catalog import catalog_documents, legacy_menu
from voice_agent.data.qdrant_rag import QdrantMenuRAG
from voice_agent.llm.tool_services import MenuToolService
from voice_agent.llm.tools import list_products, resolve_menu_item, suggest_addons


class FakeRetriever:
    def search(self, query: str, limit: int = 4):
        return [{"score": 1.0, "text": f"fake match for {query}", "metadata": {}}]


class CatalogRAGTests(unittest.TestCase):
    def test_catalog_loads_legacy_menu_shape(self):
        menu = legacy_menu()

        self.assertIn("burger", menu)
        self.assertEqual(menu["burger"]["price"], 5.99)
        self.assertIn("cheeseburger", menu["burger"]["synonyms"])
        self.assertEqual(MENU["apple pie"]["stock"], 0)

    def test_resolves_synonyms_from_seed_catalog(self):
        self.assertEqual(resolve_menu_item("diet coke"), "soda")
        self.assertEqual(resolve_menu_item("double cheeseburger"), "burger")

    def test_catalog_documents_include_offers_and_menu_items(self):
        docs = catalog_documents()
        texts = " ".join(doc["text"] for doc in docs)

        self.assertIn("Burger costs $5.99", texts)
        self.assertIn("Promo code DISCOUNT10", texts)

    def test_local_retrieval_finds_relevant_menu_context(self):
        rag = QdrantMenuRAG()
        rag.client = None

        matches = rag.search("Do you have chocolate milkshake?", limit=2)

        self.assertTrue(matches)
        self.assertIn("Shake costs $2.99", matches[0]["text"])

    def test_list_products_returns_paginated_available_products(self):
        result = list_products.invoke({"page": 1, "page_size": 2})

        self.assertEqual(result["page"], 1)
        self.assertEqual(result["page_size"], 2)
        self.assertTrue(result["has_next"])
        self.assertEqual(len(result["products"]), 2)
        self.assertNotIn("apple pie", [product["id"] for product in result["products"]])

    def test_suggest_addons_for_main_item(self):
        result = suggest_addons.invoke({"item": "burger"})
        suggestion_ids = [suggestion["id"] for suggestion in result["suggestions"]]

        self.assertIn("fries", suggestion_ids)
        self.assertIn("soda", suggestion_ids)

    def test_menu_tool_service_accepts_injected_retriever(self):
        service = MenuToolService(catalog_service=QdrantMenuRAG().catalog_service, retriever=FakeRetriever())

        context = service.search_menu_knowledge("burgers")

        self.assertEqual(context, "fake match for burgers")


if __name__ == "__main__":
    unittest.main()
