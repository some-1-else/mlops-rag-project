import streamlit as st

from src.rag import answer_question


st.set_page_config(page_title="MLOps RAG MVP", page_icon="search", layout="wide")

st.title("MLOps RAG MVP")
st.caption("Text RAG baseline over PDFs from data/raw")

question = st.text_input("Question", placeholder="Ask a question about indexed PDFs...")

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching and generating answer..."):
        try:
            result = answer_question(question.strip())
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    if result["sources"]:
        for idx, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"{idx}. {source['source']} · page {source['page']} · score {source['score']:.3f}"
            ):
                st.write(source["text"])
    else:
        st.info("No sources found.")
