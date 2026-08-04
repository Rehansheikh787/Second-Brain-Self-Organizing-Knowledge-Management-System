"""Capture pipeline — one command saves anything to raw/ with timestamp + UUID."""

import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path

from config import RAW_DIR
from utils import compute_content_hash, save_json, list_raw_captures, load_json


class DuplicateError(Exception):
    """Raised when content has already been captured."""
    pass


def capture(content: str, source_type: str, original_filename: str = None) -> str:
    """
    Validate input, generate UUID + timestamp, compute content hash,
    check for duplicates, write to raw/<uuid>.json.
    
    Returns: the UUID string of the created capture.
    Raises: ValueError for empty/invalid input, DuplicateError for hash collision.
    """
    # Validate source type
    valid_types = ["note", "link", "file"]
    if source_type not in valid_types:
        raise ValueError(f"source_type must be one of {valid_types}, got '{source_type}'")
    
    # Validate content
    if not content or not content.strip():
        raise ValueError("Content cannot be empty")
    
    # If source_type is "file"
    if source_type == "file":
        try:
            file_path = Path(content)
            if len(content) < 512 and "\n" not in content:
                if file_path.exists() and file_path.is_file():
                    try:
                        file_content = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        file_content = file_path.read_text(encoding="latin-1")
                    original_filename = file_path.name
                    content = file_content
                elif not original_filename and (file_path.suffix or "/" in content or "\\" in content):
                    raise FileNotFoundError(f"File not found: {content}")
        except FileNotFoundError:
            raise
        except Exception:
            pass
            
    # Compute content hash
    content_hash = compute_content_hash(content)
    
    # Check for duplicates
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for existing_file in list_raw_captures():
        existing_data = load_json(existing_file)
        if existing_data.get("metadata", {}).get("content_hash") == content_hash:
            raise DuplicateError(
                f"Duplicate content detected. Existing capture: {existing_data['id']}"
            )
    
    # Generate UUID and timestamp
    capture_id = str(uuid_mod.uuid4())
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    
    # Build capture data
    data = {
        "id": capture_id,
        "created": timestamp,
        "source_type": source_type,
        "content": content,
        "metadata": {
            "original_filename": original_filename,
            "char_count": len(content),
            "content_hash": content_hash
        }
    }
    
    # Write to file
    output_path = RAW_DIR / f"{capture_id}.json"
    save_json(output_path, data)
    
    return capture_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture a note, link, or file into Second Brain")
    parser.add_argument("--type", "-t", required=True, choices=["note", "link", "file"],
                        help="Type of content to capture")
    parser.add_argument("--content", "-c", required=True,
                        help="The content to capture (text, URL, or file path)")
    
    args = parser.parse_args()
    
    try:
        capture_id = capture(args.content, args.type)
        print(f"SUCCESS: Captured! ID: {capture_id}")
        print(f"         Saved to: raw/{capture_id}.json")
    except DuplicateError as e:
        print(f"WARNING: Duplicate: {e}")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
