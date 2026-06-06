import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from voice_agent.data.factories import get_menu_retriever


def main() -> None:
    retriever = get_menu_retriever()
    retriever.ensure_index()
    if retriever.client is None:
        print("Menu RAG is using JSON catalog fallback.")
    else:
        print("Menu RAG is using Qdrant vector DB.")


if __name__ == "__main__":
    main()
