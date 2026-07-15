"""Shared configuration — single source of truth for paths, thresholds, model names."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Directories ===
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
STATIC_DIR = BASE_DIR / "static"
GRAPH_JSON = BASE_DIR / "graph.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npz"

# === LLM (Groq) ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
CLASSIFY_TEMPERATURE = 0.1
ASK_TEMPERATURE = 0.3
CLASSIFY_MAX_TOKENS = 200
ASK_MAX_TOKENS = 1000

# === Embeddings ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.65
TOP_K_RETRIEVAL = 5

# === PARA Framework ===
PARA_CATEGORIES = ["Projects", "Areas", "Resources", "Archives"]
