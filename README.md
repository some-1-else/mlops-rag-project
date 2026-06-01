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
|   |-- parse_pdfs.py
|   `-- build_index.py
`-- src/
    |-- config.py
    |-- pdf_loader.py
    |-- chunking.py
    |-- vector_db.py
    |-- rag.py
    `-- tools/
        |-- pdf_parser.py
        |-- time_series.py
        |-- analytics.py
        `-- agent_tools.py
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

1. Download the configured source files:

```bash
python scripts/download_sources.py
```

PDF reports are saved to `data/raw/`. Direct CSV/XML time-series files are saved to `data/raw/time_series/` for optional analytics tools.

2. Put any extra PDF files into `data/raw`.
3. Parse PDFs into `data/processed/documents.jsonl`:

```bash
python scripts/parse_pdfs.py
```

4. Build the local vector index:

```bash
python scripts/build_index.py
```

5. Start the Streamlit app:

```bash
streamlit run app.py
```

6. Ask a question and review the answer with cited source chunks.

## Current Scope

This MVP handles text-only PDF RAG with source citations.

The source registry lives in [docs/sources_registry.md](docs/sources_registry.md).

## Optional Data & Analytics Tools

The project also includes optional helper modules under `src/tools/`:

- `pdf_parser.py`: richer PDF parsing with text blocks, images, and optional table extraction.
- `time_series.py`: connectors for CBR currency/key-rate series, Rosstat files, and custom CSV time series.
- `analytics.py`: correlations, lag analysis, OLS regression, dynamics summaries, and Plotly charts.
- `agent_tools.py`: wrappers that expose the helper modules as optional LangChain `StructuredTool` objects.

These tools are not required for the core RAG UI path:

```text
PDF -> documents.jsonl -> Chroma -> Streamlit RAG
```

See [docs/analytics_tools.md](docs/analytics_tools.md) for examples and tool details.
