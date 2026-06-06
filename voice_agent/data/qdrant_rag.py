import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from voice_agent.data.catalog import get_default_catalog_service
from voice_agent.data.interfaces import CatalogService, EmbeddingProvider

DEFAULT_COLLECTION = "drive_thru_menu_knowledge"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_TRANSFORMER_CACHE = "cache/sentence-transformers"
DEFAULT_VECTOR_SIZE = 384

logger = logging.getLogger(__name__)
_QDRANT_DISABLED_REASON: str | None = None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cache_contains_model(cache_folder: str, model_name: str) -> bool:
    """Return true when Hugging Face already has this model in the configured cache."""
    if not cache_folder:
        return False

    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        return True

    cache_path = Path(cache_folder).expanduser()
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    snapshot_dir = cache_path / model_cache_name / "snapshots"
    return snapshot_dir.is_dir() and any(snapshot_dir.iterdir())


def _disable_qdrant(reason: str) -> None:
    global _QDRANT_DISABLED_REASON
    if _QDRANT_DISABLED_REASON != reason:
        logger.info("Menu retrieval source disabled: Qdrant vector DB. reason=%s", reason)
    _QDRANT_DISABLED_REASON = reason


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, cache_folder: str, local_files_only: bool):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_folder:
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        kwargs["cache_folder"] = cache_folder
    return SentenceTransformer(model_name, **kwargs)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """SentenceTransformers embedding adapter."""

    def __init__(
        self,
        model_name: str,
        cache_folder: str = "",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only

    @property
    def model(self):
        return _load_sentence_transformer(self.model_name, self.cache_folder, self.local_files_only)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in embedding.tolist()]


class LocalMenuRetriever:
    """Keyword/synonym retrieval strategy used when Qdrant is unavailable."""

    def __init__(self, catalog_service: CatalogService) -> None:
        self.catalog_service = catalog_service

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        logger.info("Menu retrieval source: JSON catalog fallback. query=%r limit=%s", query, limit)
        query_terms = _tokenize(query)
        scored = []
        for document in self.catalog_service.catalog_documents():
            doc_terms = _tokenize(document["text"])
            overlap = len(query_terms & doc_terms)
            synonym_hits = sum(
                1
                for synonym in document["metadata"].get("synonyms", [])
                if synonym.lower() in query.lower()
            )
            score = overlap + synonym_hits * 3
            if score:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"score": float(score), "text": document["text"], "metadata": document["metadata"]}
            for score, document in scored[:limit]
        ]


class QdrantMenuRAG:
    """Small Qdrant-backed retriever for menu and offer knowledge."""

    def __init__(
        self,
        catalog_service: CatalogService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        fallback_retriever: LocalMenuRetriever | None = None,
    ) -> None:
        self.catalog_service = catalog_service or get_default_catalog_service()
        self.collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
        self.vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", str(DEFAULT_VECTOR_SIZE)))
        self.embedding_model = os.getenv("SENTENCE_TRANSFORMER_MODEL", DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        self.embedding_cache = (
            os.getenv("SENTENCE_TRANSFORMER_CACHE") or DEFAULT_SENTENCE_TRANSFORMER_CACHE
        ).strip()
        local_only_setting = os.getenv("SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY", "auto").strip().lower()
        if local_only_setting == "auto":
            self.embedding_local_only = _cache_contains_model(self.embedding_cache, self.embedding_model)
        else:
            self.embedding_local_only = local_only_setting in {"1", "true", "yes", "on"}
        self.embedding_is_cached = _cache_contains_model(self.embedding_cache, self.embedding_model)
        self.embedding_offline_mode = _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE")
        self.recreate_collection_on_mismatch = _env_truthy("QDRANT_RECREATE_COLLECTION_ON_MISMATCH")
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            self.embedding_model,
            cache_folder=self.embedding_cache,
            local_files_only=self.embedding_local_only,
        )
        self.fallback_retriever = fallback_retriever or LocalMenuRetriever(self.catalog_service)
        self.client = self._build_client()
        self._indexed = False

    def _build_client(self):
        if _QDRANT_DISABLED_REASON:
            logger.info(
                "Menu retrieval source selected: JSON catalog fallback. reason=%s",
                _QDRANT_DISABLED_REASON,
            )
            return None

        if self.embedding_local_only and not self.embedding_is_cached:
            _disable_qdrant(
                f"embedding_model_not_cached model={self.embedding_model} cache={self.embedding_cache}"
            )
            return None
        if self.embedding_offline_mode and not self.embedding_is_cached:
            _disable_qdrant(
                f"embedding_offline_and_model_not_cached model={self.embedding_model} cache={self.embedding_cache}"
            )
            return None

        try:
            from qdrant_client import QdrantClient
        except ImportError:
            logger.info("qdrant-client is not installed; using JSON catalog fallback.")
            return None

        url = (os.getenv("QDRANT_URL") or "").strip()
        api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        local_path = (os.getenv("QDRANT_LOCAL_PATH") or "cache/qdrant").strip()
        timeout = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "1.0"))
        try:
            if url:
                logger.info(
                    "Menu retrieval source available: Qdrant vector DB. mode=remote collection=%s url=%s",
                    self.collection_name,
                    url,
                )
                return QdrantClient(url=url, api_key=api_key, timeout=timeout)
            logger.info(
                "Menu retrieval source available: Qdrant vector DB. mode=local collection=%s path=%s",
                self.collection_name,
                local_path,
            )
            return QdrantClient(path=local_path)
        except Exception:
            logger.exception("Could not initialize Qdrant; using JSON catalog fallback.")
            _disable_qdrant("qdrant_init_failed")
            return None

    def _embed(self, text: str) -> list[float]:
        return self.embedding_provider.embed(text)

    def _collection_has_compatible_vectors(self) -> bool:
        if self.client is None:
            return False

        info = self.client.get_collection(self.collection_name)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            return False

        size = getattr(vectors_config, "size", None)
        distance = str(getattr(vectors_config, "distance", "")).lower()
        return int(size or 0) == self.vector_size and "cosine" in distance

    def _create_collection(self) -> None:
        from qdrant_client.http import models

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )

    def ensure_index(self) -> None:
        if self._indexed or self.client is None:
            return

        try:
            from qdrant_client.http import models

            collections = self.client.get_collections().collections
            existing = {collection.name for collection in collections}
            collection_exists = self.collection_name in existing
            if collection_exists and not self._collection_has_compatible_vectors():
                if not self.recreate_collection_on_mismatch:
                    _disable_qdrant("qdrant_collection_vector_config_mismatch")
                    self.client = None
                    return

                logger.info(
                    "Recreating Qdrant collection with compatible vector config. collection=%s vector_size=%s",
                    self.collection_name,
                    self.vector_size,
                )
                self.client.delete_collection(collection_name=self.collection_name)
                collection_exists = False

            if not collection_exists:
                logger.info(
                    "Creating Qdrant vector collection for menu retrieval. collection=%s vector_size=%s",
                    self.collection_name,
                    self.vector_size,
                )
                self._create_collection()

            points = []
            documents = self.catalog_service.catalog_documents()
            for index, document in enumerate(documents):
                points.append(
                    models.PointStruct(
                        id=index + 1,
                        vector=self._embed(document["text"]),
                        payload={
                            "doc_id": document["id"],
                            "kind": document["kind"],
                            "text": document["text"],
                            **document["metadata"],
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)
            self._indexed = True
            logger.info(
                "Menu retrieval source indexed: Qdrant vector DB. collection=%s documents=%s embedding_model=%s",
                self.collection_name,
                len(documents),
                self.embedding_model,
            )
        except Exception:
            logger.exception("Could not index menu documents in Qdrant; using JSON catalog fallback.")
            _disable_qdrant("qdrant_index_failed")
            self.client = None

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        self.ensure_index()
        if self.client is None:
            logger.info("Menu retrieval source selected: JSON catalog fallback. reason=qdrant_unavailable")
            return self.fallback_retriever.search(query, limit)

        try:
            logger.info(
                "Menu retrieval source selected: Qdrant vector DB. collection=%s query=%r limit=%s",
                self.collection_name,
                query,
                limit,
            )
            query_vector = self._embed(query)
            if hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                results = response.points
            return [
                {
                    "score": float(result.score),
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload,
                }
                for result in results
            ]
        except Exception:
            logger.exception("Qdrant search failed; using JSON catalog fallback.")
            _disable_qdrant("qdrant_search_failed")
            self.client = None
            return self.fallback_retriever.search(query, limit)


@lru_cache(maxsize=1)
def get_menu_rag() -> QdrantMenuRAG:
    return QdrantMenuRAG()


def retrieve_menu_context(query: str, limit: int = 4) -> str:
    matches = get_menu_rag().search(query, limit)
    return " ".join(match["text"] for match in matches)
