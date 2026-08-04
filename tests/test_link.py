import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_compute_embedding_returns_vector():
    from link import compute_embedding
    vec = compute_embedding("Python is a programming language")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert np.linalg.norm(vec) > 0  # not a zero vector

def test_similar_texts_have_high_similarity():
    from link import compute_embedding
    v1 = compute_embedding("Python asyncio for concurrent programming")
    v2 = compute_embedding("Async programming in Python with asyncio")
    v3 = compute_embedding("Recipe for chocolate cake with frosting")
    
    # Cosine similarity
    sim_related = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sim_unrelated = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    
    assert sim_related > sim_unrelated
    assert sim_related > 0.5

def test_link_all_notes_creates_bidirectional_links():
    from link import link_all_notes
    from utils import write_frontmatter, read_frontmatter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"
        
        # Create two related notes
        write_frontmatter(wiki_dir / "note-a.md", {
            "id": "note-a", "title": "Python asyncio", "category": "Resources",
            "tags": ["python"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "Python asyncio is used for concurrent IO operations in Python")
        
        write_frontmatter(wiki_dir / "note-b.md", {
            "id": "note-b", "title": "Async Python", "category": "Resources",
            "tags": ["python"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "Python asyncio is a library to write concurrent code using the async/await syntax in Python.")
        
        # Create one unrelated note
        write_frontmatter(wiki_dir / "note-c.md", {
            "id": "note-c", "title": "Cake recipe", "category": "Archives",
            "tags": ["cooking"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "Mix flour sugar eggs and butter for a delicious chocolate cake")
        
        with patch('link.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir), \
             patch('link.EMBEDDINGS_FILE', embeddings_file):
            
            new_links = link_all_notes()
            assert new_links > 0
            
            # Check note-a has link to note-b
            meta_a, _ = read_frontmatter(wiki_dir / "note-a.md")
            linked_ids_a = [l["id"] for l in meta_a["links"]]
            assert "note-b" in linked_ids_a
            
            # Check note-b has link to note-a (bidirectional)
            meta_b, _ = read_frontmatter(wiki_dir / "note-b.md")
            linked_ids_b = [l["id"] for l in meta_b["links"]]
            assert "note-a" in linked_ids_b

def test_compute_embeddings_batch_returns_matrix():
    from link import compute_embeddings
    texts = ["Python asyncio concurrency", "Recipe for cake"]
    matrix = compute_embeddings(texts)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 384)
