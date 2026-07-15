# Second Brain — Implementation Plan

> **For execution:** Follow this plan task-by-task. Each task is one atomic action (2–5 minutes). TDD workflow: test → fail → implement → pass → commit.

**Goal:** Build a personal knowledge management system that captures notes, auto-classifies them (PARA), auto-links via embeddings, visualizes as an interactive graph, and answers natural-language questions via RAG — deployed as a Streamlit app.

**Architecture:** Filesystem-based (Markdown + JSON), local embeddings (`all-MiniLM-L6-v2`), Groq API for LLM (classification + RAG), Cytoscape.js for graph visualization, Streamlit for UI + deployment. See [Architecture.md](file:///d:/My Projects/Massai Live Project Course/Masai Project Jully/Second Brain/docs/Architecture.md) for full details.

**Tech Stack:** Python 3.10+, Groq API, sentence-transformers, NumPy, PyYAML, Streamlit, Cytoscape.js

**Source documents:**
- [ProblemStatement.md](file:///d:/My Projects/Massai Live Project Course/Masai Project Jully/Second Brain/docs/ProblemStatement.md)
- [Architecture.md](file:///d:/My Projects/Massai Live Project Course/Masai Project Jully/Second Brain/docs/Architecture.md)

---

## Phase 0 — Project Scaffold (Pre-Week 1)

> Set up the repo structure, shared config, shared utilities, and dependencies. No features yet — just the foundation every module imports from.

---

### Task 0.1: Create directory structure

**Files:**
- Create: `raw/` (empty directory with `.gitkeep`)
- Create: `wiki/` (empty directory with `.gitkeep`)
- Create: `static/` (empty directory with `.gitkeep`)
- Create: `tests/` (empty directory with `__init__.py`)

**Step 1: Create all directories and placeholder files**

```bash
mkdir raw wiki static tests
New-Item raw/.gitkeep -ItemType File
New-Item wiki/.gitkeep -ItemType File
New-Item static/.gitkeep -ItemType File
New-Item tests/__init__.py -ItemType File
```

**Step 2: Commit**

```bash
git init
git add .
git commit -m "chore: scaffold project directory structure"
```

---

### Task 0.2: Create `requirements.txt`

**Files:**
- Create: `requirements.txt`

**Step 1: Write requirements file**

```
streamlit>=1.28.0
groq>=0.4.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```
Expected: All packages install without errors.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt with pinned dependencies"
```

---

### Task 0.3: Create `.env.example` and `.gitignore`

**Files:**
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: Write `.env.example`**

```
# Required: Get your free API key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
```

**Step 2: Write `.gitignore`**

```
.env
__pycache__/
*.pyc
.pytest_cache/
embeddings.npz
*.egg-info/
dist/
build/
.streamlit/secrets.toml
```

**Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env.example and .gitignore"
```

---

### Task 0.4: Create `config.py`

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

def test_config_paths_exist():
    from config import RAW_DIR, WIKI_DIR, STATIC_DIR, GRAPH_JSON, EMBEDDINGS_FILE
    assert isinstance(RAW_DIR, Path)
    assert isinstance(WIKI_DIR, Path)
    assert isinstance(GRAPH_JSON, Path)

def test_config_constants():
    from config import SIMILARITY_THRESHOLD, TOP_K_RETRIEVAL, PARA_CATEGORIES, EMBEDDING_MODEL
    assert 0 < SIMILARITY_THRESHOLD < 1
    assert TOP_K_RETRIEVAL > 0
    assert len(PARA_CATEGORIES) == 4
    assert "Projects" in PARA_CATEGORIES
    assert isinstance(EMBEDDING_MODEL, str)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

**Step 3: Write implementation**

```python
# config.py
"""Shared configuration — single source of truth for paths, thresholds, model names."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Directories ===
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
STATIC_DIR = BASE_DIR / "static"
GRAPH_JSON = BASE_DIR / "graph.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npz"

# === LLM (Groq) ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-70b-versatile"
CLASSIFY_TEMPERATURE = 0.1
ASK_TEMPERATURE = 0.3
CLASSIFY_MAX_TOKENS = 200
ASK_MAX_TOKENS = 1000

# === Embeddings ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.65
TOP_K_RETRIEVAL = 5

# === PARA Framework ===
PARA_CATEGORIES = ["Projects", "Areas", "Resources", "Archives"]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add config.py with shared project configuration"
```

---

### Task 0.5: Create `utils.py`

**Files:**
- Create: `utils.py`
- Test: `tests/test_utils.py`

**Step 1: Write the failing tests**

```python
# tests/test_utils.py
import json
import tempfile
from pathlib import Path

def test_compute_content_hash_deterministic():
    from utils import compute_content_hash
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    assert h1 == h2
    assert h1.startswith("sha256:")

def test_compute_content_hash_different_input():
    from utils import compute_content_hash
    h1 = compute_content_hash("hello")
    h2 = compute_content_hash("world")
    assert h1 != h2

def test_frontmatter_roundtrip():
    from utils import read_frontmatter, write_frontmatter
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.md"
        metadata = {"id": "abc", "title": "Test Note", "tags": ["a", "b"]}
        body = "This is the note body.\nWith multiple lines."
        write_frontmatter(path, metadata, body)
        read_meta, read_body = read_frontmatter(path)
        assert read_meta["id"] == "abc"
        assert read_meta["title"] == "Test Note"
        assert read_meta["tags"] == ["a", "b"]
        assert read_body.strip() == body.strip()

def test_load_save_json():
    from utils import load_json, save_json
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        data = {"key": "value", "num": 42}
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'utils'`

**Step 3: Write implementation**

```python
# utils.py
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
    """Return all .md files in wiki/ directory."""
    return sorted(WIKI_DIR.glob("*.md"))


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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: add utils.py with frontmatter parsing, hashing, and JSON I/O"
```

---

## Phase 1 — The Archivist: Capture Pipeline (Week 1)

> Build `capture.py` — one command saves anything (note, link, file) to `raw/` with timestamp + UUID. Handle edge cases: duplicates, empty input, encoding errors.

**Success criteria:**
- [ ] `raw/` and `wiki/` folder structure exists
- [ ] One command captures a note, a link, AND a file
- [ ] Every capture has a timestamp + unique ID
- [ ] Captures store source type metadata
- [ ] 10+ real items captured
- [ ] Script handles edge cases gracefully

---

### Task 1.1: Write core `capture()` function — happy path test

**Files:**
- Create: `capture.py`
- Test: `tests/test_capture.py`

**Step 1: Write the failing test**

```python
# tests/test_capture.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_capture_note_creates_json_file():
    from capture import capture
    import config
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(config, 'RAW_DIR', Path(tmpdir)):
            uuid = capture("This is a test note", "note")
            assert uuid is not None
            filepath = Path(tmpdir) / f"{uuid}.json"
            assert filepath.exists()
            data = json.loads(filepath.read_text())
            assert data["id"] == uuid
            assert data["source_type"] == "note"
            assert data["content"] == "This is a test note"
            assert "created" in data
            assert "metadata" in data
            assert data["metadata"]["content_hash"].startswith("sha256:")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_capture.py::test_capture_note_creates_json_file -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'capture'`

**Step 3: Write minimal implementation**

```python
# capture.py
"""Capture pipeline — one command saves anything to raw/ with timestamp + UUID."""

import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path

from config import RAW_DIR
from utils import compute_content_hash, save_json, list_raw_captures, load_json


class DuplicateError(Exception):
    """Raised when content has already been captured."""
    pass


def capture(content: str, source_type: str) -> str:
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
    
    # If file type, read file contents
    if source_type == "file":
        file_path = Path(content)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {content}")
        try:
            file_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            file_content = file_path.read_text(encoding="latin-1")
        original_filename = file_path.name
        content = file_content
    else:
        original_filename = None
    
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_capture.py::test_capture_note_creates_json_file -v
```
Expected: 1 PASSED

**Step 5: Commit**

```bash
git add capture.py tests/test_capture.py
git commit -m "feat: add capture() function — saves notes to raw/ with UUID + timestamp"
```

---

### Task 1.2: Test capture for all 3 source types

**Files:**
- Modify: `tests/test_capture.py`

**Step 1: Add tests for link and file capture**

Append to `tests/test_capture.py`:

```python
def test_capture_link():
    from capture import capture
    import config
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(config, 'RAW_DIR', Path(tmpdir)):
            uuid = capture("https://example.com/article", "link")
            data = json.loads((Path(tmpdir) / f"{uuid}.json").read_text())
            assert data["source_type"] == "link"
            assert data["content"] == "https://example.com/article"

def test_capture_file():
    from capture import capture
    import config
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_input.txt"
        test_file.write_text("File contents here", encoding="utf-8")
        
        raw_dir = Path(tmpdir) / "raw"
        raw_dir.mkdir()
        with patch.object(config, 'RAW_DIR', raw_dir):
            uuid = capture(str(test_file), "file")
            data = json.loads((raw_dir / f"{uuid}.json").read_text())
            assert data["source_type"] == "file"
            assert data["content"] == "File contents here"
            assert data["metadata"]["original_filename"] == "test_input.txt"
```

**Step 2: Run all tests** → Expected: 3 PASSED

**Step 3: Commit**

```bash
git add tests/test_capture.py
git commit -m "test: add capture tests for link and file source types"
```

---

### Task 1.3: Test edge cases — empty input, invalid type, duplicates

**Files:**
- Modify: `tests/test_capture.py`

**Step 1: Add edge case tests**

Append to `tests/test_capture.py`:

```python
import pytest

def test_capture_empty_content_raises():
    from capture import capture
    with pytest.raises(ValueError, match="empty"):
        capture("", "note")

def test_capture_whitespace_only_raises():
    from capture import capture
    with pytest.raises(ValueError, match="empty"):
        capture("   \n  ", "note")

def test_capture_invalid_type_raises():
    from capture import capture
    with pytest.raises(ValueError, match="source_type"):
        capture("some content", "tweet")

def test_capture_duplicate_raises():
    from capture import capture, DuplicateError
    import config
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(config, 'RAW_DIR', Path(tmpdir)):
            capture("unique content here", "note")
            with pytest.raises(DuplicateError):
                capture("unique content here", "note")

def test_capture_file_not_found_raises():
    from capture import capture
    with pytest.raises(FileNotFoundError):
        capture("/nonexistent/path/file.txt", "file")
```

**Step 2: Run all tests** → Expected: 8 PASSED

**Step 3: Commit**

```bash
git add tests/test_capture.py
git commit -m "test: add edge case tests for capture — empty input, duplicates, invalid type"
```

---

### Task 1.4: Add CLI interface to `capture.py`

**Files:**
- Modify: `capture.py` (append CLI block at the bottom)

**Step 1: Add argparse CLI**

Append to `capture.py`:

```python
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
        print(f"✅ Captured! ID: {capture_id}")
        print(f"   Saved to: raw/{capture_id}.json")
    except DuplicateError as e:
        print(f"⚠️  Duplicate: {e}")
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")
```

**Step 2: Manual test** → `python capture.py --type note --content "Test note from CLI"`

**Step 3: Commit**

```bash
git add capture.py
git commit -m "feat: add CLI interface to capture.py (--type, --content)"
```

---

### Task 1.5: Capture 10+ real items

> **Manual task.** Gather 10+ real pieces of your own scattered information and capture them using the CLI.

**Step 1:** Run `python capture.py -t note -c "..."` with YOUR real content (10+ times)

**Step 2:** Verify count: `python -c "from utils import list_raw_captures; print(f'{len(list_raw_captures())} items captured')"`

**Step 3:** Commit: `git add raw/ && git commit -m "data: capture 10+ real items into raw/"`

---

### ✅ Phase 1 Checkpoint

```bash
pytest tests/ -v
python -c "from utils import list_raw_captures; captures = list_raw_captures(); print(f'✅ {len(captures)} raw captures found'); assert len(captures) >= 10"
git tag v0.1-archivist -m "Week 1 complete: Capture Pipeline"
```

---

## Phase 2 — The Librarian: Classify + Link (Week 2)

> Build `classify.py` (PARA classification via Groq) and `link.py` (embeddings + auto-linking).

**Success criteria:**
- [ ] Any raw capture → category + tags + summary automatically
- [ ] PARA categorization working correctly
- [ ] Classified notes saved as structured Markdown with YAML frontmatter
- [ ] Embeddings computed per note
- [ ] Related notes auto-linked bidirectionally
- [ ] Similarity threshold is configurable (default: 0.65)
- [ ] Runs on 15+ real items → organized `wiki/`
- [ ] LLM errors handled gracefully

---

### Task 2.1: Create Groq LLM client helper

**Files:** Create `llm_client.py` + `tests/test_llm_client.py`

> Full code for both files is provided in the artifact version at [implementation_plan.md](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0e6919e6-a873-489d-bd1c-e9bbc07665f2/implementation_plan.md) (lines 729–870). Follow TDD: test → fail → implement → pass → commit.

**Commit:** `git commit -m "feat: add llm_client.py — Groq API wrapper with retry + JSON parsing"`

---

### Task 2.2: Write `classify_note()` function

**Files:** Create `classify.py` + `tests/test_classify.py`

> Full code at artifact lines 874–1068. Key functions: `classify_note(raw_path) -> Path` and `classify_all_pending() -> list[Path]`.

**Commit:** `git commit -m "feat: add classify.py — PARA classification via Groq LLM"`

---

### Task 2.3: Write `link.py` — embedding computation + auto-linking

**Files:** Create `link.py` + `tests/test_link.py`

> Full code at artifact lines 1072–1298. Key functions: `compute_embedding()`, `load_embeddings()`, `save_embeddings()`, `find_similar()`, `link_all_notes()`.

**Commit:** `git commit -m "feat: add link.py — embedding computation and auto-linking"`

---

### Task 2.4: Test bidirectional linking end-to-end

> Full test code at artifact lines 1302–1368. Creates 2 related notes + 1 unrelated note, asserts bidirectional links appear.

**Commit:** `git commit -m "test: add end-to-end bidirectional linking test"`

---

### Task 2.5: Run classify + link on real data

1. Create `.env` with your Groq API key
2. `python classify.py` — classifies all pending raw captures
3. `python link.py` — computes embeddings and links related notes
4. Verify: `python -c "from utils import list_wiki_notes; print(f'{len(list_wiki_notes())} wiki notes')"`
5. Commit: `git add wiki/ embeddings.npz && git commit -m "data: classify and link 15+ real notes"`

---

### ✅ Phase 2 Checkpoint

```bash
pytest tests/ -v
python -c "from utils import list_wiki_notes; assert len(list_wiki_notes()) >= 10"
git tag v0.2-librarian -m "Week 2 complete: Self-Organizing Wiki"
```

---

## Phase 3 — The Cartographer: Graph Visualization (Week 3)

> Build `build_graph.py` + `static/graph.html` (Cytoscape.js).

**Success criteria:**
- [ ] Script builds nodes + edges and exports clean `graph.json`
- [ ] Interactive force-directed graph renders
- [ ] Nodes color-coded by PARA category
- [ ] Hover reveals summary and tags
- [ ] Click opens full note content
- [ ] Drag + zoom work smoothly
- [ ] Orphan nodes displayed

---

### Task 3.1: Write `build_graph.py`

**Files:** Create `build_graph.py` + `tests/test_build_graph.py`

> Full code at artifact lines 1440–1588. Key functions: `build_graph() -> dict` and `export_graph() -> Path`.

**Commit:** `git commit -m "feat: add build_graph.py — export wiki notes as graph.json"`

---

### Task 3.2: Create `static/graph.html` — Cytoscape.js interactive graph

**Files:** Create `static/graph.html`

Write a standalone HTML file with:
- Cytoscape.js loaded from CDN
- PARA color scheme: Projects=`#F59E0B`, Areas=`#3B82F6`, Resources=`#10B981`, Archives=`#64748B`
- Node sizing: `10 + (link_count × 4)` pixels
- Edge width: `weight × 3` pixels, opacity = weight
- Hover tooltip: title + tags + summary
- Click: side panel with full content
- Dark theme background, `cose` force-directed layout
- Graph data injected as `window.GRAPH_DATA = {...}` by `app.py`

**Commit:** `git commit -m "feat: add interactive Cytoscape.js graph viewer"`

---

### Task 3.3: Build graph from real data and verify visually

1. `python build_graph.py` → generates `graph.json`
2. Open `static/graph.html` in browser to verify
3. Commit: `git add graph.json && git commit -m "data: export graph.json from real wiki notes"`

---

### ✅ Phase 3 Checkpoint

```bash
pytest tests/ -v
python -c "from utils import load_json; from pathlib import Path; g = load_json(Path('graph.json')); print(f'✅ Graph: {g[\"metadata\"][\"node_count\"]} nodes, {g[\"metadata\"][\"edge_count\"]} edges')"
git tag v0.3-cartographer -m "Week 3 complete: Living Brain"
```

---

## Phase 4 — The Oracle: RAG + Deploy (Week 4)

> Build `ask.py` (RAG pipeline) and `app.py` (Streamlit UI), then deploy.

**Success criteria:**
- [ ] `ask()` returns answers synthesized from your own notes
- [ ] Answers include source citations
- [ ] Graceful "no relevant notes" handling
- [ ] One Streamlit app with graph + search
- [ ] Deployed live with a public URL
- [ ] Full pipeline works end-to-end

---

### Task 4.1: Write `ask()` function — RAG pipeline

**Files:** Create `ask.py` + `tests/test_ask.py`

> Full code at artifact lines 1669–1882. Key function: `ask(question, top_k) -> {"answer", "sources", "confidence"}`.

**Commit:** `git commit -m "feat: add ask.py — RAG pipeline with retrieval + LLM + citations"`

---

### Task 4.2: Create `app.py` — Streamlit UI

**Files:** Create `app.py`

> Full code at artifact lines 1886–2001. Three sections: `render_stats()`, `render_graph()` (Cytoscape via iframe), `render_search()` (question + answer + sources).

**Step 1:** Write app.py

**Step 2:** Test locally: `streamlit run app.py`

**Commit:** `git commit -m "feat: add app.py — Streamlit UI with graph + search + stats"`

---

### Task 4.3: Deploy to Streamlit Cloud

1. Push to GitHub: `git push -u origin main`
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo → set `app.py` → add `GROQ_API_KEY` secret → deploy
3. Verify public URL: graph renders, search works, stats display

---

### Task 4.4: Write `README.md`

Cover: project description, setup (clone, install, `.env`), usage (capture, classify, link, graph, ask), deployment, tech stack, screenshots.

**Commit:** `git commit -m "docs: add README with setup, usage, and deployment instructions"`

---

### ✅ Phase 4 Checkpoint — Final

```bash
pytest tests/ -v
streamlit run app.py  # verify locally

# End-to-end verification:
python capture.py -t note -c "Final test note for verification"
python classify.py
python link.py
python build_graph.py
# Then ask about it in the Streamlit UI

git tag v1.0-oracle -m "Week 4 complete: Second Brain deployed"
git push --tags
```

---

## Summary

| Phase | Week | Tasks | What You Ship |
|-------|------|-------|---------------|
| **0** | Pre-1 | 5 | Project scaffold, config, utils |
| **1** | 1 | 5 | `capture.py` + 10+ real captures |
| **2** | 2 | 5 | `classify.py` + `link.py` + organized wiki |
| **3** | 3 | 3 | `build_graph.py` + `static/graph.html` + graph.json |
| **4** | 4 | 4 | `ask.py` + `app.py` + deployment + README |
| **Total** | — | **22** | **🧠 Second Brain — Live & deployed** |
