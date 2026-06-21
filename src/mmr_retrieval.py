"""
mmr_retrieval.py — диверсифицированный поиск по источникам (MMR-lite).

Проблема: обычный similarity search в Chroma может вернуть top_k результатов,
где ВСЕ они из одного и того же документа (например, cbr_financial_stability_review),
просто потому что этот документ большой и содержит много похожих по теме чанков.
Другие релевантные источники (например IMF) при этом не попадают в контекст LLM,
даже если они есть в более широком пуле (top-10, top-20).

Решение: берём более широкий пул кандидатов (например top 20), затем выбираем
top_k результатов с ограничением "не более N чанков от одного doc_id", сохраняя
сортировку по релевантности внутри каждого источника.

Это НЕ полноценный MMR (Maximal Marginal Relevance с учётом embedding-расстояния
между уже выбранными результатами), а упрощённая эвристика "diversity by source" —
для проекта такого масштаба (11 документов) этого достаточно и она в разы проще
в реализации и объяснении на защите, чем полный MMR.

Использование (вместо query_chunks в src/vector_db.py):

    from src.mmr_retrieval import query_chunks_diversified

    matches = query_chunks_diversified(
        question,
        top_k=5,
        pool_size=20,           # сколько кандидатов рассматриваем перед диверсификацией
        max_per_source=2,       # максимум чанков от одного doc_id в финальной выдаче
    )
"""

from __future__ import annotations

from collections import defaultdict

from src.vector_db import get_collection


def query_chunks_diversified(
    question: str,
    top_k: int = 5,
    pool_size: int = 20,
    max_per_source: int = 2,
) -> list[dict]:
    """
    Возвращает top_k чанков, ограничивая количество чанков от одного doc_id.

    Алгоритм:
        1. Запрашиваем pool_size кандидатов (больше чем top_k) у Chroma,
           отсортированных по убыванию релевантности (similarity score).
        2. Проходим по пулу по порядку (от самого релевантного к менее).
        3. Берём чанк, только если у его doc_id ещё не набрано max_per_source штук.
        4. Останавливаемся, когда набрали top_k чанков или кончился пул.
        5. Если после прохода пула получили МЕНЬШЕ top_k (например все источники
           уже исчерпали свой лимит) — догоняем оставшиеся места чанками
           с наивысшим score, игнорируя лимит max_per_source (лучше показать
           что-то релевантное, чем недобрать результатов).

    Args:
        question: вопрос пользователя.
        top_k: сколько чанков вернуть в итоге (то, что увидит LLM).
        pool_size: сколько кандидатов рассматривать до диверсификации.
                   Должно быть существенно больше top_k (рекомендуется 3-4x).
        max_per_source: максимум чанков от одного doc_id в финальной выдаче.

    Returns:
        Список dict: [{"text": ..., "metadata": {...}, "score": float}, ...]
        Тот же формат, что и query_chunks() — совместим с build_context().
    """
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=pool_size)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    pool = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        pool.append({
            "text": text,
            "metadata": metadata,
            "score": 1 - distance,
        })
    # pool уже отсортирован по убыванию score (так возвращает Chroma)

    selected: list[dict] = []
    source_counts: dict[str, int] = defaultdict(int)
    leftover: list[dict] = []

    for item in pool:
        doc_id = item["metadata"].get("doc_id", item["metadata"].get("source_file", "?"))
        if source_counts[doc_id] < max_per_source:
            selected.append(item)
            source_counts[doc_id] += 1
        else:
            leftover.append(item)

        if len(selected) >= top_k:
            break

    # Догоняем оставшиеся места, если диверсификация дала меньше top_k
    # (например если в пуле всего 2 разных источника)
    if len(selected) < top_k:
        for item in leftover:
            selected.append(item)
            if len(selected) >= top_k:
                break

    return selected[:top_k]
