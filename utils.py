"""Shared utilities — file I/O, frontmatter parsing, content hashing."""

import hashlib
import json
from pathlib import Path
import yaml
from config import RAW_DIR, WIKI_DIR


def compute_content_hash(content: str) -> str:
    """SHA-256 hash for duplicate detection."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def read_frontmatter(filepath: Path) -> tuple[dict, str]:
    """Parse YAML frontmatter + body from a Markdown file.
    
    Expects files in format:
    ---
    key: value
    ---
    body content
    """
    text = filepath.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    
    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return metadata, body


def write_frontmatter(filepath: Path, metadata: dict, body: str) -> None:
    """Write YAML frontmatter + body to a Markdown file."""
    frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
    content = f"---\n{frontmatter}---\n\n{body}\n"
    filepath.write_text(content, encoding="utf-8")


def list_wiki_notes() -> list[Path]:
    """Return all .md files recursively in wiki/ directory."""
    return sorted(WIKI_DIR.glob("**/*.md"))


def list_raw_captures() -> list[Path]:
    """Return all .json files in raw/ directory."""
    return sorted(RAW_DIR.glob("*.json"))


def load_json(filepath: Path) -> dict:
    """Read and parse a JSON file."""
    return json.loads(filepath.read_text(encoding="utf-8"))


def save_json(filepath: Path, data: dict) -> None:
    """Write data as formatted JSON."""
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
