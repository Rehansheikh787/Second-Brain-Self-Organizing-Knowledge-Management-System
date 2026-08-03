# Core Quality Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four core pipeline bottlenecks/fragilities: LLM JSON parsing, title extraction, batched note embeddings, and vectorized matrix linking.

**Architecture:** Update `llm_client.py` with Groq native JSON mode + regex fallback; expand `classify.py` prompt to extract explicit title; implement `compute_embeddings` batching and matrix similarity matching in `link.py`.

**Tech Stack:** Python 3.10+, Groq API (`llama-3.3-70b-versatile`), sentence-transformers (`all-MiniLM-L6-v2`), NumPy, pytest.

## Global Constraints

- **Backwards compatibility:** Existing `compute_embedding(text)` and `find_similar(...)` functions in `link.py` must remain exported with identical signatures.
- **Data scope:** New notes only. Existing wiki notes retain their current frontmatter titles.
- **Link semantics:** Exclude self-links (`note_id == target_id`), apply `SIMILARITY_THRESHOLD = 0.45`, preserve existing link merge & bidirectional logic.

---

### Task 1: Robust LLM JSON Parsing (`llm_client.py`)

**Files:**
- Modify: `llm_client.py:12-68`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: Groq API client
- Produces: `call_groq(system_prompt: str, user_content: str, temperature: float = 0.1, max_tokens: int = 200, max_retries: int = 2) -> dict`

- [ ] **Step 1: Write failing unit test for fallback regex JSON extraction**

Add a test in `tests/test_llm_client.py` verifying `call_groq` successfully extracts JSON when the response contains conversational preamble before the JSON object.

```python
def test_call_groq_extracts_json_with_preamble():
    from llm_client import call_groq
    from unittest.mock import patch, MagicMock

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = 'Here is your classification:\n{"category": "Projects", "title": "Build app", "tags": ["code"], "summary": "Build app"}'

    with patch("llm_client.Groq") as MockGroq:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockGroq.return_value = mock_client

        result = call_groq("System", "User")
        assert result["category"] == "Projects"
        assert result["title"] == "Build app"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_llm_client.py::test_call_groq_extracts_json_with_preamble -v`
Expected: FAIL with `json.JSONDecodeError`

- [ ] **Step 3: Update `call_groq` in `llm_client.py`**

Update `call_groq` to pass `response_format={"type": "json_object"}` to Groq and apply regex fallback when direct parsing fails.

```python
import json
import re
import time
import logging
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


def call_groq(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 200,
    max_retries: int = 2
) -> dict:
    """
    Send a request to Groq and parse the JSON response.
    Uses native JSON mode, with markdown fence & regex fallbacks.
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # Fallback 1: Markdown code block stripping
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            
            # Try primary parse
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                # Fallback 2: Extract first JSON object via regex
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                raise
            
        except json.JSONDecodeError:
            logger.warning(f"Attempt {attempt + 1}: Invalid JSON from LLM, retrying...")
            if attempt < max_retries:
                time.sleep(1)
                continue
            raise ValueError(f"LLM returned invalid JSON after {max_retries + 1} attempts")
            
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    
    raise ValueError("All retries exhausted")
```

- [ ] **Step 4: Run llm_client tests to verify all pass**

Run: `py -m pytest tests/test_llm_client.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add llm_client.py tests/test_llm_client.py
git commit -m "fix: add Groq native JSON mode and regex parse fallback in call_groq"
```

---

### Task 2: Real Title Extraction (`classify.py`)

**Files:**
- Modify: `classify.py:12-70`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: `call_groq`
- Produces: `classify_note(raw_path: Path) -> Path` (populates `metadata["title"]` from explicit `title` output)

- [ ] **Step 1: Update existing `test_classify_note_creates_wiki_markdown` to expect distinct title and summary**

In `tests/test_classify.py`, update `mock_llm_result` to provide both `title` and `summary` and assert `meta["title"]` equals the explicit title.

```python
        mock_llm_result = {
            "category": "Resources",
            "title": "Python Asyncio Overview",
            "tags": ["python", "asyncio", "concurrency"],
            "summary": "Python asyncio for concurrent IO operations"
        }
```

Add assertion:
```python
        assert meta["title"] == "Python Asyncio Overview"
```

- [ ] **Step 2: Run classify tests to verify failure**

Run: `py -m pytest tests/test_classify.py -v`
Expected: FAIL (`meta["title"]` equals `"Python asyncio for concurrent IO operations"` instead of `"Python Asyncio Overview"`)

- [ ] **Step 3: Update `classify.py` prompt and metadata construction**

Update `CLASSIFY_SYSTEM_PROMPT` in `classify.py`:
```python
CLASSIFY_SYSTEM_PROMPT = """You are a knowledge classifier. Categorize the note using the PARA method:
- Projects: active, time-bound goals
- Areas: ongoing responsibilities  
- Resources: reference material and interests
- Archives: completed or inactive items

Respond ONLY in valid JSON with this exact structure:
{
  "category": "one of: Projects, Areas, Resources, Archives",
  "title": "concise descriptive note title, max 80 characters",
  "tags": ["3 to 5 relevant keywords"],
  "summary": "one-line summary, max 120 characters"
}"""
```

Update title parsing in `classify_note`:
```python
    title = result.get("title") or result.get("summary") or "Untitled"
    metadata = {
        "id": note_id,
        "title": str(title)[:120],
        "category": category,
        "tags": result.get("tags", []),
        "created": raw_data["created"],
        "source_type": raw_data["source_type"],
        "links": [],
        "embedding_version": EMBEDDING_MODEL
    }
```

- [ ] **Step 4: Run classify tests to verify pass**

Run: `py -m pytest tests/test_classify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add classify.py tests/test_classify.py
git commit -m "feat: extract distinct note titles during PARA classification"
```

---

### Task 3: Batch Embeddings & Matrix Link Computation (`link.py`)

**Files:**
- Modify: `link.py:25-161`
- Test: `tests/test_link.py`

**Interfaces:**
- Consumes: `SentenceTransformer.encode(list[str])`, `load_embeddings()`, `save_embeddings()`, `list_wiki_notes()`, `read_frontmatter()`, `write_frontmatter()`
- Produces:
  - `compute_embeddings(texts: list[str]) -> np.ndarray`
  - `compute_embedding(text: str) -> np.ndarray` (calls `compute_embeddings([text])[0]`)
  - `link_all_notes() -> int` (using vectorized normalized similarity matrix)

- [ ] **Step 1: Write failing unit test for `compute_embeddings` batch function**

Add `test_compute_embeddings_batch_returns_matrix` to `tests/test_link.py`:

```python
def test_compute_embeddings_batch_returns_matrix():
    from link import compute_embeddings
    texts = ["Python asyncio concurrency", "Recipe for cake"]
    matrix = compute_embeddings(texts)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 384)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_link.py::test_compute_embeddings_batch_returns_matrix -v`
Expected: FAIL with `ImportError: cannot import name 'compute_embeddings'`

- [ ] **Step 3: Implement `compute_embeddings`, batch encoding, and matrix matmul linking in `link.py`**

Modify `link.py`:

```python
import logging
import numpy as np
from pathlib import Path

from config import EMBEDDING_MODEL, EMBEDDINGS_FILE, WIKI_DIR, SIMILARITY_THRESHOLD
from utils import read_frontmatter, write_frontmatter, list_wiki_notes

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into a (N, 384) matrix using batch processing."""
    if not texts:
        return np.array([]).reshape(0, 384)
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


def compute_embedding(text: str) -> np.ndarray:
    """Encode single text into a 384-dim vector."""
    return compute_embeddings([text])[0]


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


def link_all_notes() -> int:
    """
    Batch compute embeddings for new notes and calculate bidirectional
    links using normalized matrix dot-products.
    """
    wiki_notes = list_wiki_notes()
    if not wiki_notes:
        return 0
    
    existing_ids, existing_vectors = load_embeddings()
    id_to_index = {nid: idx for idx, nid in enumerate(existing_ids)}
    
    all_ids = []
    all_vectors_list = []
    new_texts = []
    new_indices = []
    
    # Identify new vs cached notes
    for idx, note_path in enumerate(wiki_notes):
        meta, body = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        all_ids.append(note_id)
        
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
    
    for i, note_path in enumerate(wiki_notes):
        meta, body = read_frontmatter(note_path)
        source_id = all_ids[i]
        
        # Extract row similarities, mask out self-link
        row_sims = sim_matrix[i].copy()
        row_sims[i] = -1.0
        
        # Find candidates >= SIMILARITY_THRESHOLD
        match_indices = np.where(row_sims >= SIMILARITY_THRESHOLD)[0]
        
        similar = [
            {"id": all_ids[j], "similarity": round(float(row_sims[j]), 4)}
            for j in match_indices
        ]
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        
        existing_links = {link["id"] for link in meta.get("links", [])}
        updated_links = list(meta.get("links", []))
        
        for match in similar:
            if match["id"] not in existing_links:
                updated_links.append(match)
                new_links += 1
                
        if len(updated_links) != len(meta.get("links", [])):
            meta["links"] = updated_links
            write_frontmatter(note_path, meta, body)
            
    print(f"  Processed {len(wiki_notes)} notes ({len(new_texts)} new)")
    print(f"  Created {new_links} new links")
    
    return new_links
```

- [ ] **Step 4: Run all link tests to verify pass**

Run: `py -m pytest tests/test_link.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add link.py tests/test_link.py
git commit -m "perf: implement batch embeddings and matrix similarity in link_all_notes"
```

---

### Task 4: Integration Verification Across Test Suite

**Files:**
- Test: `tests/`

- [ ] **Step 1: Execute full pytest test suite**

Run: `py -m pytest -v`
Expected: ALL PASS across test suite.

- [ ] **Step 2: Verify git status clean**

Run: `git status`
Confirm working tree clean.
