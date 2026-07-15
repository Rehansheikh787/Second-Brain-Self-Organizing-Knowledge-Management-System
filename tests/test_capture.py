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
