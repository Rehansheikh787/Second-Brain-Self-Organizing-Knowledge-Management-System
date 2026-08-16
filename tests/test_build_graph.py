import json
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_build_graph_exports_valid_structure():
    from build_graph import build_graph
    from utils import write_frontmatter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        
        # Create note A linking to note B
        write_frontmatter(res_dir / "note-a.md", {
            "id": "note-a", "title": "Note A Title", "category": "Resources",
            "tags": ["tag1"], "created": "2026-01-01", "source_type": "note",
            "links": [{"id": "note-b", "similarity": 0.85}], "embedding_version": "test"
        }, "Content of Note A")
        
        write_frontmatter(res_dir / "note-b.md", {
            "id": "note-b", "title": "Note B Title", "category": "Resources",
            "tags": ["tag2"], "created": "2026-01-01", "source_type": "note",
            "links": [{"id": "note-a", "similarity": 0.85}], "embedding_version": "test"
        }, "Content of Note B")
        
        with patch('build_graph.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir):
            
            graph = build_graph()
            
            assert "nodes" in graph
            assert "edges" in graph
            assert "metadata" in graph
            
            assert len(graph["nodes"]) == 2
            assert len(graph["edges"]) == 1  # Deduplicated bidirectional edge
            
            node_ids = [n["data"]["id"] for n in graph["nodes"]]
            assert "note-a" in node_ids
            assert "note-b" in node_ids
            
            edge = graph["edges"][0]["data"]
            assert (edge["source"] == "note-a" and edge["target"] == "note-b") or \
                   (edge["source"] == "note-b" and edge["target"] == "note-a")
            assert edge["weight"] == 0.85


def test_graph_update_after_note_deletion():
    from build_graph import build_graph, export_graph
    from utils import write_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        graph_json = Path(tmpdir) / "graph.json"

        # Create note A and note B
        note_a = res_dir / "note-a.md"
        note_b = res_dir / "note-b.md"

        write_frontmatter(note_a, {
            "id": "note-a", "title": "Note A", "category": "Resources",
            "links": [{"id": "note-b", "similarity": 0.8}]
        }, "Content A")

        write_frontmatter(note_b, {
            "id": "note-b", "title": "Note B", "category": "Resources",
            "links": [{"id": "note-a", "similarity": 0.8}]
        }, "Content B")

        with patch('build_graph.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir), \
             patch('build_graph.GRAPH_JSON', graph_json):

            # Initial graph has 2 nodes, 1 edge
            graph1 = build_graph()
            assert len(graph1["nodes"]) == 2
            assert len(graph1["edges"]) == 1

            # Delete note B
            note_b.unlink()

            # Rebuild graph
            graph2 = build_graph()
            assert len(graph2["nodes"]) == 1
            assert len(graph2["edges"]) == 0
            assert graph2["nodes"][0]["data"]["id"] == "note-a"
