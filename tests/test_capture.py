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
        with patch('capture.RAW_DIR', Path(tmpdir)), patch('utils.RAW_DIR', Path(tmpdir)):
            uuid = capture("https://example.com/article", "link")
            data = json.loads((Path(tmpdir) / f"{uuid}.json").read_text())
            assert data["source_type"] == "link"
            assert data["content"] == "https://example.com/article"

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
