"""PARA classification via LLM — auto-classify raw captures into structured wiki notes."""

import logging
from pathlib import Path

from config import WIKI_DIR, PARA_CATEGORIES, CLASSIFY_TEMPERATURE, CLASSIFY_MAX_TOKENS, EMBEDDING_MODEL
from utils import load_json, write_frontmatter, list_raw_captures, list_wiki_notes
from llm_client import call_groq

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM_PROMPT = """You are a knowledge classifier. Categorize the following note using the PARA method:
- Projects: active, time-bound goals
- Areas: ongoing responsibilities  
- Resources: reference material and interests
- Archives: completed or inactive items

Respond ONLY in valid JSON with this exact structure:
{
  "category": "one of: Projects, Areas, Resources, Archives",
  "tags": ["3 to 5 relevant keywords"],
  "summary": "one-line summary, max 120 characters"
}"""


def classify_note(raw_path: Path) -> Path:
    """
    Read raw capture, send to LLM for classification,
    parse response, write structured Markdown to wiki/<uuid>.md.
    
    Returns: Path to the created wiki note.
    """
    raw_data = load_json(raw_path)
    content = raw_data["content"]
    note_id = raw_data["id"]
    
    # Call LLM for classification
    result = call_groq(
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        user_content=content,
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS
    )
    
    # Validate and sanitize category
    category = result.get("category", "Resources")
    if category not in PARA_CATEGORIES:
        logger.warning(f"Invalid category '{category}' from LLM, defaulting to 'Resources'")
        category = "Resources"
    
    # Build frontmatter metadata
    metadata = {
        "id": note_id,
        "title": result.get("summary", "Untitled")[:120],
        "category": category,
        "tags": result.get("tags", []),
        "created": raw_data["created"],
        "source_type": raw_data["source_type"],
        "links": [],
        "embedding_version": EMBEDDING_MODEL
    }
    
    # Write wiki note
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_path = WIKI_DIR / f"{note_id}.md"
    write_frontmatter(wiki_path, metadata, content)
    
    return wiki_path


def classify_all_pending() -> list[Path]:
    """
    Find all raw captures without a corresponding wiki note,
    classify each one.
    
    Returns: list of created wiki Paths.
    """
    import time
    
    existing_wiki_ids = {p.stem for p in list_wiki_notes()}
    pending = [p for p in list_raw_captures() if p.stem not in existing_wiki_ids]
    
    created = []
    total = len(pending)
    
    for i, raw_path in enumerate(pending, 1):
        try:
            print(f"  [{i}/{total}] Classifying {raw_path.stem[:8]}...")
            wiki_path = classify_note(raw_path)
            created.append(wiki_path)
            
            # Rate limit protection
            if i < total:
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Failed to classify {raw_path.name}: {e}")
            print(f"  ⚠️  Skipped {raw_path.stem[:8]}: {e}")
            continue
    
    return created


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("🏷️  Classifying pending raw captures...")
    created = classify_all_pending()
    print(f"\n✅ Classified {len(created)} notes into wiki/")
