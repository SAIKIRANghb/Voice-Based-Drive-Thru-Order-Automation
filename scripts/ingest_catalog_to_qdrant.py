import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from voice_agent.data.factories import get_catalog_service
from voice_agent.data.qdrant_rag import QdrantMenuRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index catalog menu and offer documents into Qdrant.")
    parser.add_argument("--collection", help="Qdrant collection name. Defaults to QDRANT_COLLECTION.")
    parser.add_argument("--limit-check", default="burger", help="Query to verify retrieval after indexing.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the target collection first.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.collection:
        os.environ["QDRANT_COLLECTION"] = args.collection

    catalog_service = get_catalog_service()
    documents = catalog_service.catalog_documents()
    rag = QdrantMenuRAG(catalog_service=catalog_service)
    if args.recreate and rag.client is not None:
        collections = rag.client.get_collections().collections
        if rag.collection_name in {collection.name for collection in collections}:
            rag.client.delete_collection(collection_name=rag.collection_name)
    rag.ensure_index()

    if rag.client is None:
        raise RuntimeError("Qdrant indexing did not complete; retriever fell back to local search.")

    matches = rag.search(args.limit_check, limit=3)
    print(f"Indexed {len(documents)} documents into collection '{rag.collection_name}'.")
    print(f"Verification query: {args.limit_check!r}")
    for match in matches:
        metadata = match.get("metadata", {})
        label = metadata.get("name") or metadata.get("code") or metadata.get("doc_id") or "document"
        print(f"- {label}: score={match['score']:.4f}")


if __name__ == "__main__":
    main()
