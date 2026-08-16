import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

def test_retrieve_context_finds_relevant_notes():
    from ask import retrieve_context
    from utils import write_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"

        # Create 2 wiki notes
        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Python Virtual Environments", "category": "Resources",
            "tags": ["python", "venv"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "A virtual environment isolates Python dependencies.")

        note2_path = res_dir / "note2.md"
        write_frontmatter(note2_path, {
            "id": "note2", "title": "Baking Chocolate Cake", "category": "Resources",
            "tags": ["baking", "food"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "Mix flour, sugar, and cocoa powder to bake a cake.")

        # Create dummy embeddings
        v1 = np.ones(384, dtype=np.float32)
        v2 = np.zeros(384, dtype=np.float32)
        v2[0] = 1.0

        np.savez(embeddings_file, ids=np.array(["note1", "note2"]), vectors=np.array([v1, v2]))

        q_vec = np.ones(384, dtype=np.float32)

        with patch('ask.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir), \
             patch('ask.EMBEDDINGS_FILE', embeddings_file), \
             patch('link.EMBEDDINGS_FILE', embeddings_file), \
             patch('ask.compute_embedding', return_value=q_vec):

            retrieved = retrieve_context("How do I isolate Python dependencies?", top_k=2)

            assert len(retrieved) > 0
            assert retrieved[0]["id"] == "note1"
            assert "isolated" in retrieved[0]["content"] or "isolates" in retrieved[0]["content"]

def test_ask_synthesizes_answer_with_citations():
    from ask import ask
    from utils import write_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Python Virtual Environments", "category": "Resources",
            "tags": ["python"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "Virtual environments manage isolated package spaces.")

        v1 = np.ones(384, dtype=np.float32)
        np.savez(embeddings_file, ids=np.array(["note1"]), vectors=np.array([v1]))

        mock_llm_response = {
            "answer": "Virtual environments isolate package spaces to prevent dependency conflicts.",
            "citations": ["note1"]
        }

        with patch('ask.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir), \
             patch('ask.EMBEDDINGS_FILE', embeddings_file), \
             patch('link.EMBEDDINGS_FILE', embeddings_file), \
             patch('ask.compute_embedding', return_value=v1), \
             patch('ask.call_groq', return_value=mock_llm_response):

            res = ask("What are Python virtual environments?", top_k=1)

            assert "answer" in res
            assert "sources" in res
            assert len(res["sources"]) == 1
            assert res["sources"][0]["id"] == "note1"
            assert "Virtual environments isolate" in res["answer"]


def test_ask_with_conversation_history():
    from ask import ask
    from utils import write_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Python Virtual Environments", "category": "Resources",
            "tags": ["python"], "created": "2026-01-01", "source_type": "note",
            "links": [], "embedding_version": "test"
        }, "Virtual environments manage isolated package spaces.")

        v1 = np.ones(384, dtype=np.float32)
        np.savez(embeddings_file, ids=np.array(["note1"]), vectors=np.array([v1]))

        mock_llm_response = {
            "answer": "In simpler terms, it keeps your project tools separate.",
            "citations": ["note1"]
        }

        history = [
            {"role": "user", "content": "What is Python venv?"},
            {"role": "assistant", "content": "Virtual environments manage isolated package spaces."}
        ]

        with patch('ask.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir), \
             patch('ask.EMBEDDINGS_FILE', embeddings_file), \
             patch('link.EMBEDDINGS_FILE', embeddings_file), \
             patch('ask.compute_embedding', return_value=v1), \
             patch('ask.call_groq', return_value=mock_llm_response):

            res = ask("Explain simply", conversation_history=history, top_k=1)

            assert "answer" in res
            assert "sources" in res
            assert len(res["sources"]) == 1
            assert "separate" in res["answer"]
