import os
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, OPENAI_MODEL, ROOT_DIR
from analytics_tab import render_analytics_tab

load_dotenv(ROOT_DIR / ".env")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

CHROMA_DIR = ROOT_DIR / "vector_store" / "chroma"

SYSTEM_PROMPT = """You answer questions using ONLY the provided context below. Follow these three rules EXACTLY — pick exactly ONE of them, never mix them:

RULE 1 — Direct answer found:
If the context directly answers the question, answer it concisely and cite the source inline like [1], [2] using the bracket numbers shown in the context.

RULE 2 — No direct answer, but related information exists:
If the context does NOT directly answer the question, but contains related or partial information
(for example: a forecast instead of an actual historical figure, or data for a different period/region),
start your answer with exactly this sentence:
"Точного ответа на вопрос в документах нет, но есть смежная информация:"
Then give that related information concisely, citing the source like [1], [2].
Do NOT say "no information" and then immediately contradict yourself by giving an answer anyway —
pick RULE 2 wording from the start if the match is only partial.

RULE 3 — Nothing relevant:
If the context is completely unrelated to the question, say only:
"В загруженных документах недостаточно информации."
Do not add anything else in this case.

Keep answers concise. Always cite source_file and page for any fact you state."""

DEFAULT_TOP_K = 5
POOL_MULTIPLIER = 4    # сколько кандидатов рассматриваем перед диверсификацией (top_k * это число)
MAX_PER_SOURCE = 2     # максимум чанков от одного source_file в финальной выдаче


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


def search_chunks_raw(question: str, n_results: int) -> list[dict]:
    """Сырой запрос к Chroma без диверсификации — отсортирован по убыванию score."""
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=n_results)

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


def diversify_by_source(
    matches: list[dict],
    top_k: int,
    max_per_source: int = MAX_PER_SOURCE,
) -> list[dict]:
    """
    Ограничивает количество чанков от одного source_file в финальной выдаче.

    Проблема, которую это решает: большой документ (например, обзор на 90+ страниц)
    может занять весь top_k одними своими чанками, просто потому что он содержит
    много текста на близкую тему. Из-за этого другие релевантные источники
    (например, отчёты МВФ) не попадают в контекст LLM, хотя физически есть в индексе
    и были бы в top-10/top-20.

    Args:
        matches: результаты search_chunks_raw(), отсортированные по убыванию score.
        top_k: сколько чанков вернуть в итоге.
        max_per_source: максимум чанков от одного source_file.

    Returns:
        Диверсифицированный список длины <= top_k.
    """
    from collections import defaultdict

    selected = []
    source_counts = defaultdict(int)
    leftover = []

    for item in matches:
        source = item["metadata"].get("source_file", "?")
        if source_counts[source] < max_per_source:
            selected.append(item)
            source_counts[source] += 1
        else:
            leftover.append(item)
        if len(selected) >= top_k:
            break

    # Если диверсификация дала меньше top_k (мало разных источников в пуле) —
    # догоняем оставшиеся места следующими по релевантности, игнорируя лимит.
    if len(selected) < top_k:
        for item in leftover:
            selected.append(item)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def search_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Поиск с диверсификацией по источникам.

    Запрашивает более широкий пул кандидатов (top_k * POOL_MULTIPLIER),
    затем выбирает top_k с ограничением на количество чанков от одного файла.
    """
    pool_size = max(top_k * POOL_MULTIPLIER, top_k)
    raw_matches = search_chunks_raw(question, n_results=pool_size)
    return diversify_by_source(raw_matches, top_k=top_k, max_per_source=MAX_PER_SOURCE)


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


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MLOps RAG MVP", page_icon="search", layout="wide")

tab_rag, tab_charts = st.tabs(["🔍 RAG — поиск по документам", "📊 Text-to-Chart агент"])

with tab_charts:
    render_analytics_tab(generate_answer_fn=generate_answer)

with tab_rag:
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
    top_k = st.slider("Sources", min_value=1, max_value=100, value=DEFAULT_TOP_K)
    st.caption(
        f"Поиск диверсифицирован по источникам: не более {MAX_PER_SOURCE} чанков "
        f"от одного файла (пул кандидатов: top_k × {POOL_MULTIPLIER})."
    )

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

        with st.expander("Retrieved context / debug"):
            for idx, match in enumerate(matches, start=1):
                metadata = match["metadata"]
                st.markdown(
                    "\n".join(
                        [
                            f"**Chunk {idx}**",
                            f"- source_file: `{metadata['source_file']}`",
                            f"- page: `{metadata['page']}`",
                            f"- chunk_id: `{metadata['chunk_id']}`",
                            f"- score: `{match['score']:.3f}`",
                        ]
                    )
                )
                st.text(preview_text(match["content"], max_length=2000))