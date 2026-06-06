import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)

from qdrant_client import QdrantClient


def main() -> None:
    collection = os.getenv("QDRANT_COLLECTION", "drive_thru_menu_knowledge")
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    local_path = os.getenv("QDRANT_LOCAL_PATH", "cache/qdrant").strip()

    client = QdrantClient(url=url, api_key=api_key, timeout=5) if url else QdrantClient(path=local_path)
    info = client.get_collection(collection)
    print(info)


if __name__ == "__main__":
    main()
