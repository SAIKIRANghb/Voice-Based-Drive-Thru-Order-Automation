from functools import lru_cache

from voice_agent.data.catalog import JsonCatalogRepository, SwiggyCatalogService
from voice_agent.data.interfaces import CatalogService
from voice_agent.data.qdrant_rag import QdrantMenuRAG


@lru_cache(maxsize=1)
def get_catalog_service() -> CatalogService:
    """Singleton provider for catalog operations."""
    return SwiggyCatalogService(JsonCatalogRepository())


@lru_cache(maxsize=1)
def get_menu_retriever() -> QdrantMenuRAG:
    """Singleton provider for menu retrieval."""
    return QdrantMenuRAG(catalog_service=get_catalog_service())
