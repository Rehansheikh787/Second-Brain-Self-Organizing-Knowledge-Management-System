import json
import tempfile
from pathlib import Path

def test_compute_content_hash_deterministic():
    from utils import compute_content_hash
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    assert h1 == h2
    assert h1.startswith("sha256:")

def test_compute_content_hash_different_input():
    from utils import compute_content_hash
    h1 = compute_content_hash("hello")
    h2 = compute_content_hash("world")
    assert h1 != h2

def test_frontmatter_roundtrip():
    from utils import read_frontmatter, write_frontmatter
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.md"
        metadata = {"id": "abc", "title": "Test Note", "tags": ["a", "b"]}
        body = "This is the note body.\nWith multiple lines."
        write_frontmatter(path, metadata, body)
        read_meta, read_body = read_frontmatter(path)
        assert read_meta["id"] == "abc"
        assert read_meta["title"] == "Test Note"
        assert read_meta["tags"] == ["a", "b"]
        assert read_body.strip() == body.strip()

def test_load_save_json():
    from utils import load_json, save_json
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        data = {"key": "value", "num": 42}
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data
