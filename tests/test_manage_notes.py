import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

def test_delete_note_cascades_cleanup():
    from manage_notes import delete_note
    from utils import write_frontmatter, read_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        raw_dir = Path(tmpdir) / "raw"
        wiki_dir.mkdir()
        raw_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"
        graph_json = Path(tmpdir) / "graph.json"

        # Create raw capture and 2 wiki notes linked to each other
        raw_path = raw_dir / "note1.json"
        raw_path.write_text(json.dumps({"id": "note1", "content": "test"}))

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Note 1", "category": "Resources",
            "tags": ["test"], "links": [{"id": "note2", "similarity": 0.8}]
        }, "Body 1")

        note2_path = res_dir / "note2.md"
        write_frontmatter(note2_path, {
            "id": "note2", "title": "Note 2", "category": "Resources",
            "tags": ["test"], "links": [{"id": "note1", "similarity": 0.8}]
        }, "Body 2")

        np.savez(embeddings_file, ids=np.array(["note1", "note2"]), vectors=np.zeros((2, 384)))

        with patch("manage_notes.WIKI_DIR", wiki_dir), \
             patch("utils.WIKI_DIR", wiki_dir), \
             patch("manage_notes.RAW_DIR", raw_dir), \
             patch("utils.RAW_DIR", raw_dir), \
             patch("manage_notes.EMBEDDINGS_FILE", embeddings_file), \
             patch("link.EMBEDDINGS_FILE", embeddings_file), \
             patch("manage_notes.GRAPH_JSON", graph_json), \
             patch("build_graph.WIKI_DIR", wiki_dir), \
             patch("build_graph.GRAPH_JSON", graph_json):

            success = delete_note("note1")
            assert success is True
            assert not note1_path.exists()
            assert not raw_path.exists()

            # Verify note2 links no longer reference note1
            meta2, _ = read_frontmatter(note2_path)
            linked_ids = [l["id"] for l in meta2.get("links", [])]
            assert "note1" not in linked_ids

            # Verify embeddings array updated
            with np.load(embeddings_file, allow_pickle=True) as stored_data:
                assert "note1" not in stored_data["ids"]

def test_update_note_relocates_category():
    from manage_notes import update_note
    from utils import write_frontmatter, read_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"
        graph_json = Path(tmpdir) / "graph.json"

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Old Title", "category": "Resources",
            "tags": ["old"]
        }, "Old Body")

        np.savez(embeddings_file, ids=np.array(["note1"]), vectors=np.zeros((1, 384)))

        with patch("manage_notes.WIKI_DIR", wiki_dir), \
             patch("utils.WIKI_DIR", wiki_dir), \
             patch("manage_notes.EMBEDDINGS_FILE", embeddings_file), \
             patch("manage_notes.GRAPH_JSON", graph_json), \
             patch("link.WIKI_DIR", wiki_dir), \
             patch("link.EMBEDDINGS_FILE", embeddings_file), \
             patch("build_graph.WIKI_DIR", wiki_dir), \
             patch("build_graph.GRAPH_JSON", graph_json):

            new_path = update_note("note1", "New Title", "Projects", ["new"], "New Body")
            assert new_path.exists()
            assert "Projects" in str(new_path)
            assert not note1_path.exists()

            meta, body = read_frontmatter(new_path)
            assert meta["title"] == "New Title"
            assert meta["category"] == "Projects"
            assert meta["tags"] == ["new"]
            assert body == "New Body"

def test_get_backlinks():
    from manage_notes import get_backlinks
    from utils import write_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Target Note", "category": "Resources",
            "links": []
        }, "Target content")

        note2_path = res_dir / "note2.md"
        write_frontmatter(note2_path, {
            "id": "note2", "title": "Source Note 2", "category": "Projects",
            "links": [{"id": "note1", "similarity": 0.85}]
        }, "Source 2 content")

        note3_path = res_dir / "note3.md"
        write_frontmatter(note3_path, {
            "id": "note3", "title": "Source Note 3", "category": "Areas",
            "links": [{"id": "note1", "similarity": 0.62}]
        }, "Source 3 content")

        with patch("manage_notes.WIKI_DIR", wiki_dir), \
             patch("utils.WIKI_DIR", wiki_dir):

            backlinks = get_backlinks("note1")
            assert len(backlinks) == 2
            assert backlinks[0]["id"] == "note2"
            assert backlinks[0]["similarity"] == 0.85
            assert backlinks[1]["id"] == "note3"
            assert backlinks[1]["similarity"] == 0.62
