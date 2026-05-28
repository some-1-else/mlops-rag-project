import os
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, OPENAI_MODEL, ROOT_DIR, TOP_K


load_dotenv(ROOT_DIR / ".env")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

CHROMA_DIR = ROOT_DIR / "vector_store" / "chroma"

SYSTEM_PROMPT = """You answer questions using only the provided context.
If the answer is not in the context, say: "В загруженных документах недостаточно информации."
Keep the answer concise and cite relevant sources inline using source_file and page."""


def index_exists(index_dir: Path) -> bool:
    return index_dir.exists() and (index_dir / "chroma.sqlite3").exists()


@st.cache_resource
def get_collection():
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_function,
    )


def search_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=top_k)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "content": document,
                "metadata": metadata,
                "score": 1 - distance,
            }
        )

    return matches


def build_context(matches: list[dict]) -> str:
    blocks = []
    for idx, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] source_file: {metadata['source_file']}",
                    f"page: {metadata['page']}",
                    f"chunk_id: {metadata['chunk_id']}",
                    f"content: {match['content']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def get_llm_config() -> dict:
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env first.")
        return {
            "provider": "openai",
            "model": OPENAI_MODEL,
            "client": OpenAI(api_key=OPENAI_API_KEY),
        }

    if LLM_PROVIDER == "ollama":
        return {
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "client": OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"),
        }

    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'ollama'."
    )


def generate_answer(messages: list[dict]) -> str:
    llm = get_llm_config()
    response = llm["client"].chat.completions.create(
        model=llm["model"],
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content


def answer_question(question: str, matches: list[dict]) -> str:
    context = build_context(matches)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return generate_answer(messages)


def preview_text(text: str, max_length: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return f"{text[:max_length].rstrip()}..."


st.set_page_config(page_title="MLOps RAG MVP", page_icon="search", layout="wide")

st.title("MLOps RAG MVP")
st.caption("Question answering over the prepared Chroma index")

try:
    llm_config = get_llm_config()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"LLM provider: {llm_config['provider']} / model: {llm_config['model']}")

if not index_exists(CHROMA_DIR):
    st.error("Index not found. Run python scripts/build_index.py first.")
    st.stop()

question = st.text_input("Question", placeholder="Ask a question about indexed documents...")
top_k = st.slider("Sources", min_value=1, max_value=10, value=TOP_K)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching the index..."):
        matches = search_chunks(question.strip(), top_k=top_k)

    if not matches:
        st.warning("No relevant chunks found in the index.")
        st.stop()

    with st.spinner("Generating answer..."):
        try:
            answer = answer_question(question.strip(), matches)
        except Exception as exc:
            st.error(f"LLM request failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for idx, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        title = (
            f"{idx}. {metadata['source_file']} · page {metadata['page']} "
            f"· chunk {metadata['chunk_id']} · score {match['score']:.3f}"
        )
        with st.expander(title):
            st.write(preview_text(match["content"]))
