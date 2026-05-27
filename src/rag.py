from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from src.vector_db import query_chunks


SYSTEM_PROMPT = """You answer questions using only the provided context.
If the context is insufficient, say that you do not know.
Always keep the answer concise and cite sources inline as [source, page X]."""


def _format_context(matches: list[dict]) -> str:
    blocks = []
    for idx, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] Source: {metadata['source']}, page {metadata['page']}",
                    match["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env first.")

    matches = query_chunks(question, top_k=top_k)
    if not matches:
        return {
            "answer": "No indexed sources found. Add PDFs to data/raw and run python scripts/build_index.py.",
            "sources": [],
        }

    client = OpenAI(api_key=OPENAI_API_KEY)
    context = _format_context(matches)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
    )

    sources = []
    for match in matches:
        metadata = match["metadata"]
        sources.append(
            {
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
                "score": match["score"],
                "text": match["text"],
            }
        )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }
