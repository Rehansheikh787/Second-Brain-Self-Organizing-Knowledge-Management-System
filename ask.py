"""RAG pipeline — query user's personal knowledge base and synthesize cited answers via Groq LLM."""

import logging
from pathlib import Path
import numpy as np

from config import (
    WIKI_DIR,
    EMBEDDINGS_FILE,
    TOP_K_RETRIEVAL,
    ASK_TEMPERATURE,
    ASK_MAX_TOKENS,
    SIMILARITY_THRESHOLD
)
from utils import list_wiki_notes, read_frontmatter
from link import compute_embedding, load_embeddings
from llm_client import call_groq

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are Second Brain, an AI assistant that answers questions based on the user's personal notes.
Synthesize a concise, clear answer using ONLY the provided context notes below.
Do not invent information outside the provided notes.
If the notes do not contain enough information to answer the question, state that clearly.

Respond ONLY in valid JSON with this exact structure:
{
  "answer": "your clear synthesized response based on the context notes",
  "citations": ["id_of_note1", "id_of_note2"]
}"""


def retrieve_context(question: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Embed question, compute similarity against stored note embeddings,
    and return top-K most relevant wiki notes with content and metadata.
    """
    wiki_notes = list_wiki_notes()
    if not wiki_notes:
        return []

    # Map note IDs to file paths
    note_file_map = {}
    for note_path in wiki_notes:
        meta, _ = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        note_file_map[note_id] = note_path

    # Load stored embeddings
    stored_ids, stored_vectors = load_embeddings()
    if len(stored_ids) == 0 or stored_vectors.shape[0] == 0:
        return []

    # Filter to only IDs that currently exist in wiki
    valid_indices = [i for i, id_val in enumerate(stored_ids) if id_val in note_file_map]
    if not valid_indices:
        return []

    valid_ids = [stored_ids[i] for i in valid_indices]
    valid_vectors = stored_vectors[valid_indices]

    # Compute embedding of question
    q_vec = compute_embedding(question)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []

    vec_norms = np.linalg.norm(valid_vectors, axis=1)
    valid_mask = vec_norms > 0

    similarities = np.zeros(len(valid_ids))
    similarities[valid_mask] = np.dot(valid_vectors[valid_mask], q_vec) / (vec_norms[valid_mask] * q_norm)

    # Sort indices by similarity descending
    sorted_idx = np.argsort(similarities)[::-1]
    top_indices = sorted_idx[:top_k]

    results = []
    for idx in top_indices:
        sim_score = float(similarities[idx])
        note_id = valid_ids[idx]
        note_path = note_file_map[note_id]
        meta, body = read_frontmatter(note_path)

        results.append({
            "id": note_id,
            "title": meta.get("title", note_path.stem),
            "category": meta.get("category", "Resources"),
            "tags": meta.get("tags", []),
            "content": body,
            "similarity": round(sim_score, 4)
        })

    return results


def ask(question: str, top_k: int = TOP_K_RETRIEVAL) -> dict:
    """
    RAG Query Pipeline:
    1. Retrieve top-K context notes relevant to question.
    2. Format context and query Groq LLM.
    3. Return synthesized response with citations and confidence score.
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a valid question.",
            "sources": [],
            "confidence": 0.0
        }

    context_notes = retrieve_context(question, top_k=top_k)

    if not context_notes or context_notes[0]["similarity"] < 0.1:
        return {
            "answer": "I don't have any relevant notes in your Second Brain to answer this question.",
            "sources": [],
            "confidence": 0.0
        }

    # Format user prompt with context notes
    context_str_parts = []
    for note in context_notes:
        part = (
            f"[Note ID: {note['id']} | Title: {note['title']} | Category: {note['category']}]\n"
            f"Content: {note['content']}\n"
        )
        context_str_parts.append(part)

    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved Context Notes:\n" + "\n".join(context_str_parts)
    )

    try:
        response = call_groq(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_content=user_prompt,
            temperature=ASK_TEMPERATURE,
            max_tokens=ASK_MAX_TOKENS
        )

        answer_text = response.get("answer", "No answer generated.")
        cited_ids = set(response.get("citations", []))

        # Filter sources to cited ones, or fall back to top retrieved context
        sources = [n for n in context_notes if n["id"] in cited_ids]
        if not sources and context_notes:
            sources = context_notes[:2]

        top_confidence = context_notes[0]["similarity"] if context_notes else 0.0

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": round(top_confidence, 2)
        }

    except Exception as e:
        logger.error(f"Failed RAG query: {e}")
        return {
            "answer": f"Error generating answer: {e}",
            "sources": context_notes,
            "confidence": round(context_notes[0]["similarity"], 2) if context_notes else 0.0
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query your Second Brain using RAG")
    parser.add_argument("question", type=str, help="Question to ask your Second Brain")
    parser.add_argument("--top_k", type=int, default=TOP_K_RETRIEVAL, help="Number of notes to retrieve")

    args = parser.parse_args()
    res = ask(args.question, top_k=args.top_k)

    print("=" * 60)
    print("Q:", args.question)
    print("=" * 60)
    print("A:", res["answer"])
    print("-" * 60)
    print("Sources:")
    for src in res["sources"]:
        print(f" - [{src['category']}] {src['title']} (similarity: {src['similarity']})")
    print("=" * 60)
