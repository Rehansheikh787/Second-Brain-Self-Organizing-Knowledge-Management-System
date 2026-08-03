# Core Quality Fixes — Design Spec

**Date:** 2026-08-03
**Status:** Approved
**Scope:** Four reliability/performance fixes to the Second Brain pipeline

## Summary

Fix four reliability and performance gaps in the Second Brain core pipeline:
LLM JSON parsing fragility, missing real-title extraction, slow per-note embedding,
and O(n²) linking with heavy constants. New notes only — existing wiki data untouched.

## 1. LLM JSON Robustness (`llm_client.py`)

**Problem:** `call_groq` parses LLM output by manual markdown-fence stripping
(`llm_client.py:43-49`). Any output that isn't cleanly fenced fails JSON parse,
consuming a retry.

**Fix:**
- Add `response_format={"type": "json_object"}` to the Groq `chat.completions.create`
  call. Groq's native JSON mode eliminates most parse failures.
- Keep the existing markdown-fence stripping as a fallback.
- Add a second fallback: regex to extract the first `{...}` JSON object if the
  fence fallback fails.
- Retry logic unchanged.

**Verification:** `tests/test_llm_client.py` passes. Mocks return plain JSON, so
existing tests cover the new code path.

## 2. Real Title Extraction (`classify.py`)

**Problem:** `classify_note` sets `title = result.get("summary")` (`classify.py:54`).
The one-line summary becomes the note title. Poor graph labels, weak RAG citations.

**Fix:**
- Extend `CLASSIFY_SYSTEM_PROMPT` to request a separate `title` field:
  ```json
  {"title": "...", "category": "...", "tags": ["..."], "summary": "..."}
  ```
- In `classify_note`, use `result.get("title", result.get("summary", "Untitled"))`.
  Graceful fallback if the LLM omits title.
- **New notes only.** Existing wiki notes keep their current titles. No re-classify
  backfill, no extra Groq API calls.

**Verification:** `tests/test_classify.py` mock updated to include `title`;
test asserts title lands in frontmatter and is distinct from summary.

## 3. Batch Embeddings (`link.py`)

**Problem:** `link_all_notes` embeds each new note with a separate
`model.encode(text)` call (`link.py:119-124`). Per-call overhead dominates with
many new notes.

**Fix:**
- Add `compute_embeddings(texts: list[str]) -> np.ndarray` that runs a single
  `model.encode(texts)` batch call.
- `link_all_notes` collects all new-note texts and encodes them in one batch.
- Keep `compute_embedding(text)` (single-text) — `ask.py` and tests depend on it.

**Verification:** `tests/test_link.py` passes; batch path returns same-shaped
vectors as the single path.

## 4. Linking Performance (`link.py`)

**Problem:** `link_all_notes` is O(n²) with heavy constants:
- `existing_ids.index(note_id)` is O(n) per lookup inside the loop.
- `find_similar` recomputes `np.linalg.norm(vectors, axis=1)` over ALL vectors on
  every call — O(n) norms done n times.
- One dot-product scan per note.

**Fix:**
- Build an `id → index` dict once, eliminating `existing_ids.index()` O(n) lookups.
- Precompute vector norms once.
- Replace the per-note `find_similar` loop with a single normalized similarity
  matrix: `sim = (V / ‖V‖) @ (V / ‖V‖)ᵀ`, then threshold each row.
  One matmul instead of n scans.
- Memory: n² floats. n=500 → ~1MB, n=2000 → ~16MB. Fine for a personal brain.
- Keep `find_similar` exported for API compatibility; no longer used internally.

**Verification:** `tests/test_link.py` `test_link_all_notes_creates_bidirectional_links`
passes — same output contract (bidirectional links, deduped edges), faster path.

## Out of Scope

- Re-classification backfill of existing notes
- Async, GPU, persistence layer
- Any UI changes
