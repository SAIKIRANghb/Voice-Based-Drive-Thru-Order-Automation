import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE = "cache/sentence-transformers"


def main() -> None:
    load_dotenv()

    model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", DEFAULT_MODEL)
    cache_folder = os.getenv("SENTENCE_TRANSFORMER_CACHE", DEFAULT_CACHE)

    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=False)
    print(f"Downloaded SentenceTransformer model '{model_name}' into '{cache_folder}'.")


if __name__ == "__main__":
    main()
