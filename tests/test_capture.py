import json
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_capture_note_creates_json_file():
    from capture import capture
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('capture.RAW_DIR', Path(tmpdir)), patch('utils.RAW_DIR', Path(tmpdir)):
            uuid = capture("This is a test note", "note")
            assert uuid is not None
            filepath = Path(tmpdir) / f"{uuid}.json"
            assert filepath.exists()
            data = json.loads(filepath.read_text())
            assert data["id"] == uuid
            assert data["source_type"] == "note"
            assert data["content"] == "This is a test note"
            assert "created" in data
            assert "metadata" in data
            assert data["metadata"]["content_hash"].startswith("sha256:")

def test_capture_link():
    from capture import capture
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('capture.RAW_DIR', Path(tmpdir)), \
             patch('utils.RAW_DIR', Path(tmpdir)), \
             patch('capture.fetch_and_clean_web_article', return_value={
                 "title": "Example Article",
                 "content": "# Example Article\nSource URL: https://example.com/article\n\nArticle body content here.",
                 "url": "https://example.com/article"
             }):
            uuid = capture("https://example.com/article", "link")
            data = json.loads((Path(tmpdir) / f"{uuid}.json").read_text())
            assert data["source_type"] == "link"
            assert "# Example Article" in data["content"]
            assert "Article body content here." in data["content"]

def test_fetch_and_clean_web_article_fallback():
    from capture import fetch_and_clean_web_article
    with patch('requests.get', side_effect=Exception("Connection error")):
        res = fetch_and_clean_web_article("https://nonexistent-site-test.org/article")
        assert "title" in res
        assert "content" in res
        assert "nonexistent-site-test.org" in res["content"]

def test_capture_file():
    from capture import capture
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_input.txt"
        test_file.write_text("File contents here", encoding="utf-8")
        
        with patch('capture.RAW_DIR', Path(tmpdir)), patch('utils.RAW_DIR', Path(tmpdir)):
            uuid = capture(str(test_file), "file")
            data = json.loads((Path(tmpdir) / f"{uuid}.json").read_text())
            assert data["source_type"] == "file"
            assert data["content"] == "File contents here"
            assert data["metadata"]["original_filename"] == "test_input.txt"

import pytest

def test_capture_empty_content_raises():
    from capture import capture
    with pytest.raises(ValueError, match="empty"):
        capture("", "note")

def test_capture_whitespace_only_raises():
    from capture import capture
    with pytest.raises(ValueError, match="empty"):
        capture("   \n  ", "note")

def test_capture_invalid_type_raises():
    from capture import capture
    with pytest.raises(ValueError, match="source_type"):
        capture("some content", "tweet")

def test_capture_duplicate_raises():
    from capture import capture, DuplicateError
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('capture.RAW_DIR', Path(tmpdir)), patch('utils.RAW_DIR', Path(tmpdir)):
            capture("unique content here", "note")
            with pytest.raises(DuplicateError):
                capture("unique content here", "note")

def test_capture_file_not_found_raises():
    from capture import capture
    with pytest.raises(FileNotFoundError):
        capture("/nonexistent/path/file.txt", "file")

def test_clean_extracted_pdf_text():
    from capture import clean_extracted_pdf_text
    raw = "You \n \n are \n \n an \n \n experienced \n \n Product- \n Manager."
    cleaned = clean_extracted_pdf_text(raw)
    assert "You are an experienced ProductManager." in cleaned
