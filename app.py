import os
from pathlib import Path
from uuid import uuid4

import chromadb
import pandas as pd
import requests
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, OPENAI_MODEL, ROOT_DIR

load_dotenv(ROOT_DIR / ".env")

try:
    from src.tools.time_series import get_cbr_key_rate, get_cbr_currency
    from src.tools.analytics import (
        compute_correlation,
        compute_lag_analysis,
        compute_dynamics,
        plot_series,
        plot_correlation,
        plot_lag_analysis,
    )
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_BASE_URL = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
GIGACHAT_AUTH_URL = os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() == "true"

CHROMA_DIR = ROOT_DIR / "vector_store" / "chroma"

SYSTEM_PROMPT = """Use the provided context to answer the question.
If the context contains partial information, provide the best possible answer and mention uncertainty.
If the context is clearly unrelated, say: "В загруженных документах недостаточно информации."
Keep the answer concise and cite relevant sources inline using source_file and page."""

DEFAULT_TOP_K = 3

SERIES_OPTIONS = {
    "CBR key rate": "cbr_key_rate",
    "USD/RUB": "R01235",
    "EUR/RUB": "R01239",
}

ANALYSIS_OPTIONS = [
    "dynamics",
    "correlation",
    "lag analysis",
    "plot series",
    "plot correlation",
    "plot lag analysis",
]


def load_series(series_key: str, start: str, end: str):
    """Загружает ряд по ключу."""
    if series_key == "cbr_key_rate":
        return get_cbr_key_rate(start=start, end=end)
    else:
        return get_cbr_currency(series_id=series_key, start=start, end=end)


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


def search_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
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

    if LLM_PROVIDER == "gigachat":
        return {
            "provider": "gigachat",
            "model": GIGACHAT_MODEL,
        }

    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'ollama'."
    )


@st.cache_data(ttl=1500)
def get_gigachat_token() -> str:
    if not GIGACHAT_AUTH_KEY:
        raise RuntimeError("GIGACHAT_AUTH_KEY is missing. Add it to .env first.")

    st.write(f"[DEBUG] GigaChat OAuth: auth_url={GIGACHAT_AUTH_URL}, provider=gigachat, verify_ssl={GIGACHAT_VERIFY_SSL}")

    try:
        response = requests.post(
            GIGACHAT_AUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid4()),
                "Authorization": f"Basic {GIGACHAT_AUTH_KEY}",
            },
            data={"scope": GIGACHAT_SCOPE},
            timeout=30,
            verify=GIGACHAT_VERIFY_SSL,
        )
        st.write(f"[DEBUG] OAuth response status: {response.status_code}")

        if response.status_code != 200:
            raise RuntimeError(f"GigaChat OAuth failed: status={response.status_code}, response={response.text[:200]}")

        return response.json()["access_token"]
    except requests.exceptions.SSLError as exc:
        raise RuntimeError("SSL certificate verification failed. GigaChat may require installing Russian trusted certificates or configuring verify=False (set GIGACHAT_VERIFY_SSL=false in .env).")


def call_gigachat(messages: list[dict]) -> str:
    token = get_gigachat_token()
    url = f"{GIGACHAT_BASE_URL.rstrip('/')}/chat/completions"
    st.write(f"[DEBUG] GigaChat Chat: base_url={GIGACHAT_BASE_URL}, model={GIGACHAT_MODEL}")

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": GIGACHAT_MODEL,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=60,
            verify=GIGACHAT_VERIFY_SSL,
        )
        st.write(f"[DEBUG] Chat response status: {response.status_code}")

        if response.status_code != 200:
            raise RuntimeError(f"GigaChat chat failed: status={response.status_code}, response={response.text[:200]}")

        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.SSLError as exc:
        raise RuntimeError("SSL certificate verification failed. GigaChat may require installing Russian trusted certificates or configuring verify=False (set GIGACHAT_VERIFY_SSL=false in .env).")


def generate_answer(messages: list[dict]) -> str:
    llm = get_llm_config()

    if llm["provider"] == "gigachat":
        return call_gigachat(messages)

    response = llm["client"].chat.completions.create(
        model=llm["model"],
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content


def answer_question(question: str, matches: list[dict]) -> tuple[str, str, list[dict]]:
    context = build_context(matches)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    answer = generate_answer(messages)
    return answer, context, messages


def preview_text(text: str, max_length: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return f"{text[:max_length].rstrip()}..."


st.set_page_config(page_title="MLOps RAG MVP", page_icon="search", layout="wide")

mode = st.radio(
    "Режим",
    ["RAG over documents", "Analytics demo"],
    horizontal=True,
    index=0,
)

if mode == "Analytics demo":
    if not ANALYTICS_AVAILABLE:
        st.error("Analytics modules not available. Install: pandas scipy statsmodels plotly")
        st.stop()

    st.title("Analytics Demo")
    st.caption("Анализ временных рядов ЦБ РФ")

    col1, col2, col3 = st.columns(3)
    with col1:
        series1_name = st.selectbox("Series 1", options=list(SERIES_OPTIONS.keys()), index=0)
    with col2:
        series2_name = st.selectbox("Series 2", options=list(SERIES_OPTIONS.keys()), index=1)
    with col3:
        analysis_type = st.selectbox("Analysis", options=ANALYSIS_OPTIONS, index=1)

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("Start", value=pd.Timestamp("2022-01-01"))
    with col_date2:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Loading data..."):
            try:
                s1_key = SERIES_OPTIONS[series1_name]
                s2_key = SERIES_OPTIONS[series2_name]

                df1 = load_series(s1_key, str(start_date), str(end_date))
                df2 = load_series(s2_key, str(start_date), str(end_date))

                st.success(f"Loaded: {len(df1)} records ({series1_name}), {len(df2)} records ({series2_name})")

                if analysis_type == "dynamics":
                    result = compute_dynamics(df1)
                    st.subheader("Dynamics Summary")
                    st.write(result.summary())

                    st.subheader(f"Data: {series1_name}")
                    st.dataframe(df1.head(10))

                elif analysis_type == "correlation":
                    try:
                        result = compute_correlation(df1, df2)
                        st.subheader("Correlation Result")
                        st.write(result.summary())

                        st.subheader("Data (first 10 rows)")
                        st.dataframe(df1.head(10))
                    except Exception as e:
                        st.error(f"Correlation failed: {e}")

                elif analysis_type == "lag analysis":
                    try:
                        result = compute_lag_analysis(df1, df2, max_lag=12)
                        st.subheader("Lag Analysis Result")
                        st.write(result.summary())

                        st.subheader("Data (first 10 rows)")
                        st.dataframe(df1.head(10))
                    except Exception as e:
                        st.error(f"Lag analysis failed: {e}")

                elif analysis_type == "plot series":
                    st.subheader("Plot: Series")
                    try:
                        plot_path = plot_series(df1, df2, title=f"{series1_name} vs {series2_name}")
                        st.success(f"Chart saved: {plot_path}")

                        with open(plot_path, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                    except Exception as e:
                        st.error(f"Plot failed: {e}")

                elif analysis_type == "plot correlation":
                    st.subheader("Plot: Correlation")
                    try:
                        result = compute_correlation(df1, df2)
                        st.write(result.summary())

                        plot_path = plot_correlation(df1, df2, result=result, title=f"Correlation: {series1_name} vs {series2_name}")
                        st.success(f"Chart saved: {plot_path}")

                        with open(plot_path, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                    except Exception as e:
                        st.error(f"Correlation plot failed: {e}")

                elif analysis_type == "plot lag analysis":
                    st.subheader("Plot: Lag Analysis")
                    try:
                        result = compute_lag_analysis(df1, df2, max_lag=12)
                        st.write(result.summary())

                        plot_path = plot_lag_analysis(result, title=f"Lag Analysis: {series1_name} vs {series2_name}")
                        st.success(f"Chart saved: {plot_path}")

                        with open(plot_path, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=600, scrolling=True)
                    except Exception as e:
                        st.error(f"Lag analysis plot failed: {e}")

            except Exception as e:
                st.error(f"Error: {e}")

    st.stop()

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
top_k = st.slider("Sources", min_value=1, max_value=10, value=DEFAULT_TOP_K)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching the index..."):
        matches = search_chunks(question.strip(), top_k=top_k)

    if not matches:
        st.warning("No relevant chunks found in the index.")
        st.stop()

    with st.spinner("Generating answer..."):
        try:
            answer, context, messages = answer_question(question.strip(), matches)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error("LLM request failed. Please try again.")
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        st.text(context)

    with st.expander("LLM Prompt Debug"):
        for msg in messages:
            st.text(f"{msg['role'].upper()}:\n{msg['content'][:1000]}")
            st.divider()

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
