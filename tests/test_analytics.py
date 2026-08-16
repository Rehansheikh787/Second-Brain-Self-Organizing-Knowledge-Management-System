import tempfile
from pathlib import Path
from unittest.mock import patch

def test_get_analytics_data_calculates_metrics():
    from analytics import get_analytics_data
    from utils import write_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        wiki_dir = Path(tmpdir) / "wiki"
        raw_dir.mkdir()
        wiki_dir.mkdir()

        # Create dummy raw capture
        (raw_dir / "raw1.json").write_text('{"id": "raw1"}', encoding="utf-8")

        # Create dummy wiki notes
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        write_frontmatter(res_dir / "note1.md", {
            "id": "note1", "title": "Python Concurrency", "category": "Resources",
            "tags": ["python", "async"], "created": "2026-08-01T10:00:00+00:00",
            "links": [{"id": "note2", "similarity": 0.8}]
        }, "Content 1")

        write_frontmatter(res_dir / "note2.md", {
            "id": "note2", "title": "Python Asyncio", "category": "Resources",
            "tags": ["python"], "created": "2026-08-02T10:00:00+00:00",
            "links": [{"id": "note1", "similarity": 0.8}]
        }, "Content 2")

        with patch('analytics.list_raw_captures', return_value=list(raw_dir.glob("*.json"))), \
             patch('analytics.list_wiki_notes', return_value=list(wiki_dir.glob("**/*.md"))):

            data = get_analytics_data()

            assert data["raw_count"] == 1
            assert data["wiki_count"] == 2
            assert data["total_links"] == 1
            assert data["categories"]["Resources"] == 2
            assert len(data["top_connected"]) == 2
            assert data["top_tags"][0][0] == "python"
