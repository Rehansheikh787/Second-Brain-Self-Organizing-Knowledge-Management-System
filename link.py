"""Embedding computation & auto-linking — find related notes and link them bidirectionally."""

import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDINGS_FILE, WIKI_DIR, SIMILARITY_THRESHOLD
from utils import read_frontmatter, write_frontmatter, list_wiki_notes

logger = logging.getLogger(__name__)

# Lazy-loaded model (only loads when first called)
_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def compute_embedding(text: str) -> np.ndarray:
    """Encode text into a 384-dim vector using the configured model."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def load_embeddings() -> tuple[list[str], np.ndarray]:
    """Load stored embeddings from embeddings.npz. Returns (ids, vectors)."""
    if not EMBEDDINGS_FILE.exists():
        return [], np.array([]).reshape(0, 384)
    
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    ids = data["ids"].tolist()
    vectors = data["vectors"]
    return ids, vectors


def save_embeddings(ids: list[str], vectors: np.ndarray) -> None:
    """Persist embeddings to embeddings.npz."""
    np.savez(
        EMBEDDINGS_FILE,
        ids=np.array(ids),
        vectors=vectors,
        model=EMBEDDING_MODEL
    )


def find_similar(
    query_vector: np.ndarray,
    ids: list[str],
    vectors: np.ndarray,
    exclude_id: str = None,
    threshold: float = None
) -> list[dict]:
    """
    Find all notes with cosine similarity >= threshold to the query vector.
    Returns: [{"id": "...", "similarity": 0.78}, ...] sorted by similarity desc.
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    
    if len(ids) == 0:
        return []
    
    # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
    norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    
    # Avoid division by zero
    valid = norms > 0
    similarities = np.zeros(len(ids))
    if query_norm > 0:
        similarities[valid] = np.dot(vectors[valid], query_vector) / (norms[valid] * query_norm)
    
    results = []
    for i, (note_id, sim) in enumerate(zip(ids, similarities)):
        if note_id == exclude_id:
            continue
        if sim >= threshold:
            results.append({"id": note_id, "similarity": round(float(sim), 4)})
    
    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def link_all_notes() -> int:
    """
    Compute/update embeddings for all wiki notes,
    find similar pairs, write bidirectional links.
    
    Returns: number of new links created.
    """
    wiki_notes = list_wiki_notes()
    if not wiki_notes:
        return 0
    
    # Load existing embeddings
    existing_ids, existing_vectors = load_embeddings()
    existing_set = set(existing_ids)
    
    # Find new notes that need embedding
    all_ids = []
    all_vectors_list = []
    new_count = 0
    
    for note_path in wiki_notes:
        meta, body = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        
        if note_id in existing_set:
            # Reuse existing embedding
            idx = existing_ids.index(note_id)
            all_ids.append(note_id)
            all_vectors_list.append(existing_vectors[idx])
        else:
            # Compute new embedding
            text = f"{meta.get('title', '')} {body}"
            vec = compute_embedding(text)
            all_ids.append(note_id)
            all_vectors_list.append(vec)
            new_count += 1
    
    all_vectors = np.array(all_vectors_list)
    
    # Save updated embeddings
    save_embeddings(all_ids, all_vectors)
    
    # Find and write bidirectional links
    new_links = 0
    
    for i, note_path in enumerate(wiki_notes):
        meta, body = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        
        similar = find_similar(
            all_vectors[i], all_ids, all_vectors,
            exclude_id=note_id
        )
        
        # Get existing link IDs
        existing_links = {link["id"] for link in meta.get("links", [])}
        
        # Merge new links (keep existing, add new)
        updated_links = list(meta.get("links", []))
        for match in similar:
            if match["id"] not in existing_links:
                updated_links.append(match)
                new_links += 1
        
        # Write back if links changed
        if len(updated_links) != len(meta.get("links", [])):
            meta["links"] = updated_links
            write_frontmatter(note_path, meta, body)
    
    print(f"  Computed embeddings for {new_count} new notes")
    print(f"  Created {new_links} new links")
    
    return new_links


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Computing embeddings and linking related notes...")
    new_links = link_all_notes()
    print(f"\nDone! {new_links} new links created.")
