"""Embedding computation & auto-linking — find related notes and link them bidirectionally."""

import logging
import numpy as np
from pathlib import Path

from config import EMBEDDING_MODEL, EMBEDDINGS_FILE, WIKI_DIR, SIMILARITY_THRESHOLD
from utils import read_frontmatter, write_frontmatter, list_wiki_notes

logger = logging.getLogger(__name__)

# Lazy-loaded model (only loads when first called)
_model = None


class FallbackEmbedder:
    """Fallback vectorizer using 384-dimensional HashingVectorizer with n-grams."""
    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vectorizer = HashingVectorizer(n_features=384, norm='l2', alternate_sign=False, ngram_range=(1, 2))

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([]).reshape(0, 384)
        X = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return X if len(texts) > 1 else X[0] if not isinstance(texts, list) else X


def _get_model():
    """Lazy-load the embedding model with fallback."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable ({e}), using FallbackEmbedder")
            _model = FallbackEmbedder()
    return _model


def invalidate_note_embedding(note_id: str) -> None:
    """Purge a note's vector from embeddings.npz so it gets recomputed on next update."""
    stored_ids, stored_vectors = load_embeddings()
    if note_id in stored_ids:
        idx = stored_ids.index(note_id)
        new_ids = [nid for i, nid in enumerate(stored_ids) if i != idx]
        new_vectors = np.delete(stored_vectors, idx, axis=0) if stored_vectors.shape[0] > 0 else stored_vectors
        save_embeddings(new_ids, new_vectors)


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into a (N, 384) matrix using batch processing."""
    if not texts:
        return np.array([]).reshape(0, 384)
    model = _get_model()
    res = model.encode(texts, convert_to_numpy=True)
    if isinstance(res, list):
        res = np.array(res, dtype=np.float32)
    return res


def compute_embedding(text: str) -> np.ndarray:
    """Encode single text into a 384-dim vector."""
    res = compute_embeddings([text])
    return res[0] if len(res) > 0 else np.zeros(384, dtype=np.float32)


def load_embeddings() -> tuple[list[str], np.ndarray]:
    """Load stored embeddings from embeddings.npz. Returns (ids, vectors)."""
    if not EMBEDDINGS_FILE.exists():
        return [], np.array([]).reshape(0, 384)
    
    with np.load(EMBEDDINGS_FILE, allow_pickle=True) as data:
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
    Retained for API compatibility.
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    
    if len(ids) == 0:
        return []
    
    norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    
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


import re

STOP_WORDS = {
    "what", "are", "is", "the", "a", "an", "and", "or", "in", "of", "to", "for",
    "with", "how", "do", "does", "can", "why", "where", "this", "that", "it",
    "from", "have", "note", "document", "file", "using", "your", "my", "you",
    "was", "were", "been", "being", "which", "when", "who", "whom", "will", "would"
}


def extract_domain_keywords(text: str) -> set[str]:
    """Extract meaningful non-stopword tokens (min length 3)."""
    words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower()))
    return words - STOP_WORDS


def link_all_notes() -> int:
    """
    Batch compute embeddings for new notes and calculate bidirectional
    links using normalized matrix dot-products with domain keyword filtering
    and a top-3 links per note cap.
    """
    wiki_notes = list_wiki_notes()
    if not wiki_notes:
        return 0
    
    existing_ids, existing_vectors = load_embeddings()
    id_to_index = {nid: idx for idx, nid in enumerate(existing_ids)}
    
    all_ids = []
    all_vectors_list = []
    note_data = []
    new_texts = []
    new_indices = []
    
    # Identify new vs cached notes
    for idx, note_path in enumerate(wiki_notes):
        meta, body = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        all_ids.append(note_id)
        keywords = extract_domain_keywords(f"{meta.get('title', '')} {body}")
        note_data.append({"id": note_id, "path": note_path, "meta": meta, "body": body, "keywords": keywords})
        
        if note_id in id_to_index:
            old_idx = id_to_index[note_id]
            all_vectors_list.append(existing_vectors[old_idx])
        else:
            text = f"{meta.get('title', '')} {body}"
            new_texts.append(text)
            new_indices.append(idx)
            all_vectors_list.append(None)  # Placeholder
            
    # Batch encode new notes in single call
    if new_texts:
        new_vecs = compute_embeddings(new_texts)
        for sub_idx, orig_idx in enumerate(new_indices):
            all_vectors_list[orig_idx] = new_vecs[sub_idx]
            
    all_vectors = np.array(all_vectors_list, dtype=np.float32)
    save_embeddings(all_ids, all_vectors)
    
    N = len(all_ids)
    if N <= 1:
        return 0
        
    # Matrix similarity calculation: (V / ||V||) @ (V / ||V||).T
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9  # Avoid divide by zero
    norm_vectors = all_vectors / norms
    sim_matrix = np.dot(norm_vectors, norm_vectors.T)
    
    new_links = 0
    
    for i, note in enumerate(note_data):
        source_id = note["id"]
        source_kw = note["keywords"]
        
        # Extract row similarities, mask out self-link
        row_sims = sim_matrix[i].copy()
        row_sims[i] = -1.0
        
        # Find candidate matches >= threshold with keyword verification
        candidates = []
        for j in range(N):
            if j == i:
                continue
            sim = float(row_sims[j])
            cand_kw = note_data[j]["keywords"]
            shared_kw = source_kw & cand_kw
            
            # Require either high vector similarity (>=0.55) or decent similarity (>=0.40) + shared domain keyword
            if sim >= 0.58 or (sim >= 0.38 and len(shared_kw) >= 1):
                candidates.append({"id": note_data[j]["id"], "similarity": round(sim, 4)})
                
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_links = candidates[:3]  # Cap to top 3 most relevant links
        
        meta = note["meta"]
        meta["links"] = top_links
        write_frontmatter(note["path"], meta, note["body"])
        new_links += len(top_links)
            
    print(f"  Processed {len(wiki_notes)} notes ({len(new_texts)} new)")
    print(f"  Established {new_links} total precision links")
    
    return new_links


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Computing embeddings and linking related notes...")
    new_links = link_all_notes()
    print(f"\nDone! {new_links} new links created.")
