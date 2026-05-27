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

Add your OpenAI API key to `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Put PDF files into `data/raw`.
2. Build the local vector index:

```bash
python scripts/build_index.py
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Ask a question and review the answer with cited source chunks.

## Current Scope

This MVP handles text-only PDF RAG with source citations. Tables, charts, time-series tools, and a richer UI are intentionally left for later iterations.
