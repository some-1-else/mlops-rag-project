# MLOps RAG Project

Minimal text RAG baseline for PDF documents.

## Project Structure

```text
mlops-rag-project/
|-- app.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- README.md
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- vector_store/
|-- scripts/
|   `-- build_index.py
`-- src/
    |-- config.py
    |-- pdf_loader.py
    |-- chunking.py
    |-- vector_db.py
    `-- rag.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## LLM Provider Setup

The project supports two LLM modes:

- `ollama`: local free model running on your machine
- `openai`: OpenAI API

Configuration is stored in a local `.env` file. The `.env` file is not committed to git. The repository only keeps `.env.example`.

The active provider is selected with `LLM_PROVIDER`. If you change `.env`, restart Streamlit so the app reloads the configuration.

### Ollama

Install and start Ollama on macOS:

```bash
brew install ollama
brew services start ollama
ollama pull llama3.1:8b
```

Example `.env` for Ollama:

```env
LLM_PROVIDER=ollama
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### OpenAI

Example `.env` for OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Usage

1. Put PDF files into `data/raw`.
2. Parse PDFs into `data/processed/documents.jsonl`:

```bash
python scripts/parse_pdfs.py
```

3. Build the local vector index:

```bash
python scripts/build_index.py
```

4. Start the Streamlit app:

```bash
streamlit run app.py
```

5. Ask a question and review the answer with cited source chunks.

## Current Scope

This MVP handles text-only PDF RAG with source citations. Tables, charts, time-series tools, and a richer UI are intentionally left for later iterations.
