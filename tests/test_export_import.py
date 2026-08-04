import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_extract_file_content_text():
    from export_import import extract_file_content

    mock_file = MagicMock()
    mock_file.name = "sample.txt"
    mock_file.read.return_value = b"Hello world text"

    content, file_type = extract_file_content(mock_file)
    assert content == "Hello world text"
    assert file_type == "text"

def test_extract_file_content_media_saves_file():
    from export_import import extract_file_content

    with tempfile.TemporaryDirectory() as tmpdir:
        static_dir = Path(tmpdir) / "static"
        static_dir.mkdir()

        mock_file = MagicMock()
        mock_file.name = "photo.png"
        mock_file.read.return_value = b"\x89PNG fake data"

        with patch("export_import.STATIC_DIR", static_dir):
            content, file_type = extract_file_content(mock_file)

            saved_path = static_dir / "uploads" / "photo.png"
            assert saved_path.exists()
            assert file_type == "media"
            assert "photo.png" in content
            assert "![" in content

def test_generate_zip_backup_creates_valid_zip():
    from export_import import generate_zip_backup

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir(parents=True)
        (res_dir / "note.md").write_text("Note content")

        static_dir = Path(tmpdir) / "static"
        uploads_dir = static_dir / "uploads"
        uploads_dir.mkdir(parents=True)
        (uploads_dir / "photo.png").write_text("fake image")

        with patch("export_import.WIKI_DIR", wiki_dir), \
             patch("export_import.STATIC_DIR", static_dir):

            zip_bytes = generate_zip_backup()
            assert isinstance(zip_bytes, bytes)
            assert len(zip_bytes) > 0

            # Verify zip archive contents
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                names = zf.namelist()
                assert any("note.md" in n for n in names)
                assert any("photo.png" in n for n in names)
