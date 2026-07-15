# Second Brain — Problem Statement

## The Problem

Every notes app fails the same way: you capture hundreds of notes, bookmarks, PDFs, and ideas — and then you never find them again. Information goes in, but nothing comes back out. Notes sit in folders nobody re-reads. Bookmarks pile up unread. Knowledge doesn't compound.

## The Goal

Build an end-to-end system where you can:

1. **Capture** anything (a note, a link, a file)
2. **Classify** it automatically using AI (PARA method)
3. **Auto-link** it to related knowledge via embeddings
4. **Visualize** it as a live, interactive, explorable graph
5. **Query** it in plain English and get answers synthesized from your own accumulated knowledge
6. **Deploy** it to a public URL anyone can open

> Not a notes app. Not a chatbot. A brain that organizes itself and answers for you.

---

## System Architecture (High-Level Flow)

```
Capture any note / link / file
         ↓
AI classifies & files it (PARA method)
         ↓
AI auto-links it to related notes (embeddings)
         ↓
Everything renders as a live, interactive, hoverable graph
         ↓
Ask it anything in plain English → answer pulled from YOUR notes
         ↓
Deployed on a public URL anyone can open
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.10+ | Ecosystem fit for AI/ML tooling |
| LLM (Classification & QA) | Groq API (Llama 3 / Llama 3.1) | Free tier, fast inference |
| Embeddings | `sentence-transformers` (local) | Free, runs offline, no API key needed |
| Vector Similarity | `numpy` cosine similarity (or FAISS for scale) | Lightweight for prototype, scalable later |
| Graph Visualization | Cytoscape.js *or* vis-network | Force-directed layout, hover/drag/zoom |
| Web UI | Streamlit | Rapid prototyping, built-in deploy path |
| Deployment | Streamlit Cloud *or* Hugging Face Spaces | Free hosting, public URL |
| Storage | Local filesystem (Markdown files) | Simple, git-friendly, no DB setup |

---

## Week-by-Week Problem Statements

> Each week is a self-contained problem. Build it, test it on **real data** (your own notes — not test data), and each week's output becomes the next week's input.

---

### Week 1 — The Archivist: "Capture Everything, Lose Nothing"

#### Problem

You have no single place to put things. Ideas, links, and notes scatter across apps, browser tabs, and your memory. Build the foundation: one command that captures anything into one place.

#### What to Build

1. **Set up the project structure from scratch:**
   - `raw/` — where every raw capture lands
   - `wiki/` — (used later) organized, linked notes

2. **Write a Python capture script (`capture.py`)** that takes any note, link, or file and saves it into `raw/` with:
   - A timestamp (ISO 8601 format, e.g. `2026-07-08T13:00:00`)
   - A unique ID (UUID v4)
   - The raw content (text body, URL, or file path reference)
   - Source type metadata (`note`, `link`, or `file`)

3. **Test it on 10+ real pieces** of your own scattered information.

#### Edge Cases to Handle

- Duplicate captures (same URL or identical content)
- Empty or malformed input
- Non-UTF-8 file content
- Very long notes (>10,000 chars)

#### Deliverable — "Ship the Capture Pipeline"

- A working capture script — one command saves anything to `raw/` with timestamp + unique ID.
- Your `raw/` folder populated with 10+ real captured items (not test data).
- 🏅 **Badge: The Archivist**

#### Acceptance Criteria

- [ ] `raw/` and `wiki/` folder structure exists
- [ ] One command captures a note, a link, AND a file
- [ ] Every capture has a timestamp + unique ID
- [ ] Captures store source type metadata (`note` / `link` / `file`)
- [ ] 10+ real items captured
- [ ] Script handles edge cases gracefully (empty input, duplicates)

---

### Week 2 — The Librarian: "Teach AI to Organize For You"

#### Problem

A pile of raw captures is still a mess. Manual tagging never happens. Make the AI do the filing — and make it notice when two notes are about the same thing and link them automatically.

#### What to Build

##### 2.1 — Auto-Classify (The Sorting Hat) → `classify.py`

- Write a function that sends any raw capture to a free LLM (Groq / Llama 3) and gets back:
  - A **category** using the PARA framework:
    - **P**rojects — active, time-bound goals
    - **A**reas — ongoing responsibilities
    - **R**esources — reference material and interests
    - **A**rchives — completed or inactive items
  - **Tags** (3–5 relevant keywords)
  - A **one-line summary** (≤ 120 characters)
- Run it across last week's real captures and watch them organize themselves.
- Save classified notes as structured Markdown files in `wiki/` with YAML frontmatter:

```yaml
---
id: <uuid>
title: <summary>
category: <PARA category>
tags: [tag1, tag2, tag3]
created: <timestamp>
source_type: <note|link|file>
links: []
---
<original content>
```

##### 2.2 — Auto-Link Related Notes (Connect the Dots) → `link.py`

- Compute embeddings for each note using `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, local + free).
- Compare each new capture against existing notes in `wiki/`.
- When content is related (cosine similarity ≥ **0.65 threshold** — tunable), auto-insert a bidirectional link between them.
- No manual tagging — the system notices relationships on its own.
- Store link references in the note's YAML frontmatter `links: []` field.

#### Edge Cases to Handle

- LLM returns malformed JSON → retry with fallback prompt
- Embedding model download on first run (handle gracefully)
- Notes linking to themselves
- Threshold too low → noisy links; too high → no links (make configurable)

#### Deliverable — "Ship the Self-Organizing Wiki"

- A pipeline that auto-classifies raw captures with PARA and auto-links related notes.
- Run on 15+ real items → an organized `wiki/` folder with linked notes.
- 🏅 **Badge: The Librarian**

#### Acceptance Criteria

- [ ] Any raw capture → category + tags + summary automatically
- [ ] PARA categorization working correctly
- [ ] Classified notes saved as structured Markdown with YAML frontmatter
- [ ] Embeddings computed per note (using `sentence-transformers`)
- [ ] Related notes auto-linked bidirectionally (no manual tagging)
- [ ] Similarity threshold is configurable (default: 0.65)
- [ ] Runs on 15+ real items → organized `wiki/`
- [ ] LLM errors handled gracefully (retries, fallbacks)

---

### Week 3 — The Cartographer: "Visualize the Brain"

#### Problem

Your knowledge is now organized and linked — but you can't *see* it. Turn the wiki into something you can actually look at, explore, and watch think.

#### What to Build

##### 3.1 — Graph Data Model (Give It a Shape) → `build_graph.py`

- Write a script that reads every note in `wiki/` and its links from the YAML frontmatter.
- Build a nodes-and-edges representation:
  - Every note → a **node** (with `id`, `label`, `category`, `summary`)
  - Every relationship/link → an **edge** (with `source`, `target`, `similarity_score`)
- Export it as clean JSON → `graph.json`:

```json
{
  "nodes": [
    { "id": "uuid-1", "label": "Note title", "category": "Resources", "summary": "..." }
  ],
  "edges": [
    { "source": "uuid-1", "target": "uuid-2", "weight": 0.82 }
  ]
}
```

##### 3.2 — Interactive Graph (The Brain Comes Alive)

- Use a JS graph library (Cytoscape.js or vis-network) to render:
  - Notes as **nodes** — color-coded by PARA category, sized by connection count
  - Links as **edges** — thickness proportional to similarity score
  - **Hover popups** that reveal each note's summary and tags
  - **Drag-to-explore** and **zoom** (mouse wheel + pinch)
  - **Click-to-expand** — clicking a node shows full note content in a side panel
- A force-directed graph of your own knowledge.

#### Edge Cases to Handle

- Orphan nodes (notes with no links) — still display, don't hide
- Very large graphs (100+ nodes) — consider clustering or level-of-detail
- Mobile responsiveness for the graph view

#### Deliverable — "Ship the Living Brain"

- Your wiki converted to a graph and rendered as an interactive visual brain (hover, drag, zoom), built from your real notes.
- 🏅 **Badge: The Cartographer**

#### Acceptance Criteria

- [ ] Script builds nodes + edges from wiki notes and exports clean `graph.json`
- [ ] Interactive force-directed graph renders from that JSON
- [ ] Nodes are color-coded by PARA category
- [ ] Hover reveals note summary and tags
- [ ] Click opens full note content
- [ ] Drag + zoom work smoothly
- [ ] Orphan nodes are displayed (not hidden)
- [ ] Built from your real notes, not dummy data

---

### Week 4 — The Oracle: "Ask It Anything, Ship It Public"

#### Problem

A visual brain is beautiful, but the real payoff is answers. Wire up natural-language search over everything you know — then package the whole thing into one deployable product.

#### What to Build

##### 4.1 — Ask Your Brain (Natural Language Search) → `ask.py`

- Build a single `ask(question: str) -> str` function that combines:
  - **Retrieval** — embeddings find the top-K most relevant notes (default K=5)
  - **Context assembly** — retrieved note contents are assembled into a prompt
  - **Generation** — an LLM (Groq / Llama 3) synthesizes an answer from retrieved notes
- This is **RAG (Retrieval-Augmented Generation)** over your own knowledge.
- The answer must **cite which notes** it drew from (by title or ID).
- Test against real questions about your own captured notes.

##### 4.2 — UI, Deployment, Public URL (Give It a Face) → `app.py`

- Assemble everything into one Streamlit app with two main views:
  - 📊 **Brain Graph** — the interactive knowledge graph (embed via `streamlit.components.v1.html`)
  - 🔍 **Ask Your Brain** — a search bar + answer display with source citations
- Add a **capture form** — quick-add new notes directly from the UI (optional but recommended).
- Deploy to a free platform (Streamlit Cloud or HF Spaces).
- Get a public URL anyone can open.

#### Edge Cases to Handle

- Question with no relevant notes → respond honestly ("I don't have notes on that topic")
- Very vague questions → ask for clarification or return top related notes
- LLM hallucination — ensure answers only cite actual notes, not fabricated content
- Rate limiting on free LLM tier

#### Deliverable — "Ship Second Brain" (the final product)

Deploy the complete system — capture → auto-classify → auto-link → live interactive graph → ask-anything search — all wired into one Streamlit app with a public URL.

- 🏅 **Badge: The Oracle**

#### Acceptance Criteria

- [ ] `ask()` returns answers synthesized from your own notes (retrieval + LLM)
- [ ] Answers include source citations (which notes were used)
- [ ] Graceful handling of "no relevant notes found" scenarios
- [ ] One Streamlit app contains both the graph and the search bar
- [ ] Deployed live with a public URL
- [ ] Full pipeline works end-to-end in the deployed app

---

## Final Deliverables (Whole Project)

- [ ] Public GitHub repo with a clean README + setup instructions
- [ ] Live deployed URL — interactive graph + ask-your-brain search, both working
- [ ] End-to-end flow verified: capture → classify → link → graph → ask
- [ ] All 4 weekly milestones complete (Capture Pipeline, Self-Organizing Wiki, Living Brain, Second Brain deployment)
- [ ] `.env.example` with required API keys documented
- [ ] `requirements.txt` with pinned dependency versions

---

## Repo Structure

```
second-brain/
├── raw/                    # Week 1: raw captures (timestamp + unique ID)
├── wiki/                   # Week 2: classified + auto-linked notes (Markdown + YAML)
├── static/                 # Week 3: graph HTML/JS/CSS assets
│   └── graph.html          # Interactive graph page
├── capture.py              # Week 1: one-command capture
├── classify.py             # Week 2: PARA classification via LLM
├── link.py                 # Week 2: embeddings + auto-linking
├── build_graph.py          # Week 3: nodes/edges → graph.json
├── graph.json              # Week 3: exported graph data
├── ask.py                  # Week 4: retrieval + LLM answer (RAG)
├── app.py                  # Week 4: Streamlit UI (graph + search)
├── config.py               # Shared configuration (thresholds, model names, paths)
├── utils.py                # Shared utilities (file I/O, frontmatter parsing)
├── .env.example            # Required environment variables
├── requirements.txt        # Pinned dependencies
└── README.md               # Setup instructions + project overview
```

---

## Build Order

| Step | File(s) | Week | What to Do |
|------|---------|------|------------|
| 1 | Project scaffold | — | Create repo structure, `requirements.txt`, `.env.example`, `config.py` |
| 2 | `capture.py` | 1 | Capture script → test on 10+ real items |
| 3 | `classify.py` | 2.1 | PARA classification via LLM → structured Markdown in `wiki/` |
| 4 | `link.py` | 2.2 | Embeddings + similarity auto-linking |
| 5 | `build_graph.py` | 3.1 | Parse wiki → JSON nodes/edges |
| 6 | `static/graph.html` | 3.2 | Interactive graph with Cytoscape.js / vis-network |
| 7 | `ask.py` | 4.1 | RAG pipeline: retrieval + LLM answer with citations |
| 8 | `app.py` | 4.2 | Streamlit app combining graph + search + (optional) capture |
| 9 | Deploy | 4.2 | Streamlit Cloud / HF Spaces → public URL |
| 10 | `README.md` | — | Write README, push to GitHub |

---

## Key Decisions & Assumptions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage format | Markdown + YAML frontmatter | Human-readable, git-friendly, no DB dependency |
| Embedding model | `all-MiniLM-L6-v2` | Good quality, small (80MB), runs locally |
| Similarity threshold | 0.65 (configurable) | Balanced — avoids noisy links while catching real relationships |
| LLM provider | Groq (free tier) | Fast inference, generous free limits |
| Graph library | Cytoscape.js | Better API for styling, good docs, active community |
| Deployment | Streamlit Cloud | Zero-config deploy, free tier available |
| Project name | Second Brain | Descriptive, aligns with Tiago Forte's "Second Brain" concept |