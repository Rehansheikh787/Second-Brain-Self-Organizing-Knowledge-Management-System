from pathlib import Path

def test_config_paths_exist():
    from config import RAW_DIR, WIKI_DIR, STATIC_DIR, GRAPH_JSON, EMBEDDINGS_FILE
    assert isinstance(RAW_DIR, Path)
    assert isinstance(WIKI_DIR, Path)
    assert isinstance(GRAPH_JSON, Path)

def test_config_constants():
    from config import SIMILARITY_THRESHOLD, TOP_K_RETRIEVAL, PARA_CATEGORIES, EMBEDDING_MODEL
    assert 0 < SIMILARITY_THRESHOLD < 1
    assert TOP_K_RETRIEVAL > 0
    assert len(PARA_CATEGORIES) == 4
    assert "Projects" in PARA_CATEGORIES
    assert isinstance(EMBEDDING_MODEL, str)
