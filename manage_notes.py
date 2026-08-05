"""Note lifecycle operations — delete note with cascading cleanup, update note with category relocation."""

import logging
from pathlib import Path
import numpy as np

from config import WIKI_DIR, RAW_DIR, EMBEDDINGS_FILE, GRAPH_JSON, PARA_CATEGORIES
from utils import list_wiki_notes, read_frontmatter, write_frontmatter, load_json, save_json
from link import load_embeddings, save_embeddings, link_all_notes
from build_graph import export_graph

logger = logging.getLogger(__name__)


def get_backlinks(note_id: str) -> list[dict]:
    """
    Find all incoming backlinks referencing note_id.
    Returns list of dicts: [{'id': str, 'title': str, 'category': str, 'similarity': float}]
    """
    backlinks = []
    wiki_notes = list_wiki_notes()
    
    for note_path in wiki_notes:
        meta, _ = read_frontmatter(note_path)
        source_id = meta.get("id", note_path.stem)
        if source_id == note_id:
            continue
            
        links = meta.get("links", [])
        for link in links:
            if link.get("id") == note_id:
                backlinks.append({
                    "id": source_id,
                    "title": meta.get("title", note_path.stem),
                    "category": meta.get("category", "Resources"),
                    "similarity": link.get("similarity", 0.0)
                })
                break
                
    backlinks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    return backlinks


def delete_note(note_id: str) -> bool:
    """
    Delete note by ID across wiki/, raw/, embeddings.npz, cross-links, and graph.json.
    Returns True if deleted, False if note not found.
    """
    wiki_notes = list_wiki_notes()
    target_path = None
    
    for note_path in wiki_notes:
        meta, _ = read_frontmatter(note_path)
        if meta.get("id") == note_id or note_path.stem == note_id:
            target_path = note_path
            break
            
    if not target_path or not target_path.exists():
        logger.warning(f"Note {note_id} not found for deletion.")
        return False
        
    # 1. Delete wiki markdown file
    target_path.unlink()
    
    # 2. Delete raw capture JSON if exists
    raw_path = RAW_DIR / f"{note_id}.json"
    if raw_path.exists():
        raw_path.unlink()
        
    # 3. Clean embeddings array
    stored_ids, stored_vectors = load_embeddings()
    if note_id in stored_ids:
        idx = stored_ids.index(note_id)
        new_ids = [nid for i, nid in enumerate(stored_ids) if i != idx]
        new_vectors = np.delete(stored_vectors, idx, axis=0) if stored_vectors.shape[0] > 0 else stored_vectors
        save_embeddings(new_ids, new_vectors)
        
    # 4. Clean cross-references in remaining wiki notes
    remaining_notes = list_wiki_notes()
    for note_path in remaining_notes:
        meta, body = read_frontmatter(note_path)
        links = meta.get("links", [])
        filtered_links = [l for l in links if l.get("id") != note_id]
        if len(filtered_links) != len(links):
            meta["links"] = filtered_links
            write_frontmatter(note_path, meta, body)
            
    # 5. Export fresh graph.json
    export_graph(GRAPH_JSON)
    logger.info(f"Successfully deleted note {note_id} everywhere.")
    return True


def update_note(
    note_id: str,
    new_title: str,
    new_category: str,
    new_tags: list[str],
    new_body: str
) -> Path:
    """
    Update note title, category, tags, and body.
    Relocates file to wiki/<new_category>/<note_id>.md if category changes.
    Returns Path to updated note.
    """
    if new_category not in PARA_CATEGORIES:
        new_category = "Resources"
        
    wiki_notes = list_wiki_notes()
    target_path = None
    target_meta = {}
    
    for note_path in wiki_notes:
        meta, body = read_frontmatter(note_path)
        if meta.get("id") == note_id or note_path.stem == note_id:
            target_path = note_path
            target_meta = meta
            break
            
    if not target_path or not target_path.exists():
        raise FileNotFoundError(f"Note {note_id} not found for editing.")
        
    old_category = target_meta.get("category", "Resources")
    
    # Update metadata
    target_meta["title"] = str(new_title).strip()[:120]
    target_meta["category"] = new_category
    target_meta["tags"] = [str(t).strip() for t in new_tags if str(t).strip()]
    
    dest_dir = WIKI_DIR / new_category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{note_id}.md"
    
    # If category changed and path is different, remove old file
    if target_path.resolve() != dest_path.resolve():
        if target_path.exists():
            target_path.unlink()
            
    write_frontmatter(dest_path, target_meta, new_body.strip())
    
    # Refresh embeddings, linking & graph
    link_all_notes()
    export_graph(GRAPH_JSON)
    
    logger.info(f"Successfully updated note {note_id} at {dest_path}")
    return dest_path
