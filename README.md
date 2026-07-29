# 🧠 Second Brain — Self-Organizing Knowledge Management System

A personal knowledge management system that captures notes, auto-classifies them using the **PARA framework** via Groq LLM (`llama-3.3-70b-versatile`), computes bidirectional semantic links via sentence-transformers (`all-MiniLM-L6-v2`), visualizes notes as an interactive Cytoscape.js graph, and answers natural language questions with source citations via RAG — deployed as a Streamlit web application.

---

## 🌟 Architecture & Core Modules

The system is built as a 4-phase autonomous knowledge engine:

```
[Raw Inputs] ➔ (1. The Archivist) ➔ raw/*.json
                     │
                     ▼
             (2. The Librarian) ➔ wiki/{PARA}/*.md + embeddings.npz
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
(3. The Cartographer)    (4. The Oracle)
  graph.json / Graph UI    RAG Q&A / Streamlit Web App
```

1. **The Archivist (`capture.py`)**: Unified ingestion CLI that captures raw text notes, web URLs, and files into structured JSON files with UTC timestamps, UUIDs, and SHA-256 duplicate detection.
2. **The Librarian (`classify.py` & `link.py`)**: 
   - `classify.py`: Uses Groq API (`llama-3.3-70b-versatile`) to classify notes into the PARA categories (*Projects*, *Areas*, *Resources*, *Archives*) and generate tags/summaries saved in YAML frontmatter.
   - `link.py`: Computes 384-dimensional embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) and automatically inserts bidirectional markdown frontmatter links between semantically similar notes.
3. **The Cartographer (`build_graph.py` & `static/graph.html`)**: Parses frontmatter links and exports node/edge networks to `graph.json`. Renders a dark glassmorphism interactive force-directed graph powered by **Cytoscape.js** with node scaling, search filtering, and drawer previews.
4. **The Oracle (`ask.py` & `app.py`)**:
   - `ask.py`: RAG query engine that retrieves top-$K$ context notes using embedding cosine similarity and synthesizes grounded answers with citations using Groq LLM.
   - `app.py`: Streamlit Web Dashboard integrating KPI metrics, interactive graph iframe, RAG Q&A, and in-app quick capture.

---

## 🛠️ Tech Stack

- **Language & Runtime**: Python 3.10+ / 3.12
- **LLM Engine**: Groq API (`llama-3.3-70b-versatile`)
- **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Operations**: NumPy
- **Graph Visualization**: Cytoscape.js, HTML5/CSS3
- **Web Framework**: Streamlit
- **Testing**: pytest

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Environment Setup
Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/your-username/second-brain.git
cd second-brain

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💻 Command Reference

### 📥 1. Capture Knowledge (`capture.py`)
Capture a raw text note, URL link, or local file into `raw/`:

```powershell
# Capture a text note
python capture.py -t note -c "Python virtual environments isolate project dependencies."

# Capture a web link
python capture.py -t link -c "https://streamlit.io/"

# Capture a file
python capture.py -t file -c "docs/notes.txt"
```

### 🏷️ 2. Classify Notes (`classify.py`)
Auto-classify raw captures into PARA categories (`wiki/Projects`, `wiki/Areas`, `wiki/Resources`, `wiki/Archives`):

```powershell
python classify.py
```

### 🔗 3. Compute Embeddings & Auto-Link (`link.py`)
Generate 384-dim embeddings and link related notes bidirectionally:

```powershell
python link.py
```

### 🕸️ 4. Export Knowledge Graph (`build_graph.py`)
Build node/edge network data (`graph.json` and `static/graph_data.js`):

```powershell
python build_graph.py
```

### 🤖 5. Ask Questions via RAG (`ask.py`)
Query your Second Brain from the terminal:

```powershell
python ask.py "What are Python virtual environments?"
```

### 📊 6. Summary Report (`summary.py`)
Print a database summary table across categories and links:

```powershell
python summary.py
```

### 🌐 7. Launch Web Dashboard (`app.py`)
Start the interactive Streamlit Web App:

```powershell
streamlit run app.py
```

---

## 🧪 Testing

Run the full pytest suite (20 unit & integration tests):

```powershell
pytest tests/ -v
```

---

## ☁️ Deployment to Streamlit Cloud

1. Push your repository to GitHub:
   ```bash
   git push origin main
   ```
2. Log into [share.streamlit.io](https://share.streamlit.io) and click **New App**.
3. Select your repository, branch (`main`), and set Main file path to `app.py`.
4. Go to **Advanced settings... -> Secrets** and paste your API key:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy!**

---

## 📁 Repository Structure

```
Second Brain/
├── raw/                       # Captured JSON objects with UUID + timestamps
├── wiki/                      # Categorized Markdown notes with YAML frontmatter
│   ├── Projects/
│   ├── Areas/
│   ├── Resources/
│   └── Archives/
├── static/                    # Frontend Cytoscape.js interactive graph
│   ├── graph.html
│   └── graph_data.js
├── tests/                     # Pytest suite (20 unit tests)
│   ├── test_capture.py
│   ├── test_classify.py
│   ├── test_config.py
│   ├── test_llm_client.py
│   ├── test_utils.py
│   ├── test_build_graph.py
│   └── test_ask.py
├── .streamlit/                # Streamlit configuration & theme
│   └── config.toml
├── config.py                  # Single source of truth for paths & constants
├── utils.py                   # File I/O, frontmatter parser, SHA-256 hashing
├── capture.py                 # Ingestion pipeline & CLI
├── llm_client.py              # Groq API wrapper with retry logic
├── classify.py                # LLM PARA categorization
├── link.py                    # SentenceTransformers embedding & auto-linking
├── build_graph.py             # Graph data exporter
├── ask.py                     # RAG retrieval & Q&A pipeline
├── summary.py                 # Database summary helper
├── app.py                     # Streamlit Web App Dashboard
├── requirements.txt           # Pinned dependencies
└── README.md                  # System documentation
```

---

## 📜 License
Distributed under the MIT License.
