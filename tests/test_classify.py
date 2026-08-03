import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_classify_note_creates_wiki_markdown():
    from classify import classify_note
    import config
    
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        wiki_dir = Path(tmpdir) / "wiki"
        raw_dir.mkdir()
        wiki_dir.mkdir()
        
        # Create a raw capture
        raw_data = {
            "id": "test-uuid-123",
            "created": "2026-07-08T13:00:00+05:30",
            "source_type": "note",
            "content": "Python asyncio is great for concurrent IO",
            "metadata": {"content_hash": "sha256:abc", "char_count": 42, "original_filename": None}
        }
        raw_path = raw_dir / "test-uuid-123.json"
        raw_path.write_text(json.dumps(raw_data))
        
        # Mock LLM response
        mock_llm_result = {
            "category": "Resources",
            "title": "Python Asyncio Overview",
            "tags": ["python", "asyncio", "concurrency"],
            "summary": "Python asyncio for concurrent IO operations"
        }
        
        with patch('classify.WIKI_DIR', wiki_dir), \
             patch('utils.WIKI_DIR', wiki_dir), \
             patch('utils.RAW_DIR', raw_dir), \
             patch("classify.call_groq", return_value=mock_llm_result):
            
            wiki_path = classify_note(raw_path)
            
            assert wiki_path.exists()
            assert wiki_path.name == "test-uuid-123.md"
            
            from utils import read_frontmatter
            meta, body = read_frontmatter(wiki_path)
            assert meta["category"] == "Resources"
            assert meta["tags"] == ["python", "asyncio", "concurrency"]
            assert meta["title"] == "Python Asyncio Overview"
            assert meta["id"] == "test-uuid-123"
            assert "asyncio" in body
