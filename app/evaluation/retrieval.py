from dataclasses import dataclass

from app.domain import RetrievalResult


@dataclass(frozen=True)
class EvalDocument:
    """评测用文档。

    第一版先让每篇文档只产生一个 chunk，用 source_uri 判断是否召回正确资料。
    后续可以升级为 chunk_id 级别标注。
    """

    title: str
    source_uri: str
    content: str


@dataclass(frozen=True)
class EvalQuestion:
    """评测问题和人工标注答案。"""

    query: str
    expected_source_uris: set[str]


@dataclass(frozen=True)
class RetrievalEvalReport:
    """召回率评测结果。"""

    total_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


def recall_at_k(runs: list[tuple[EvalQuestion, list[RetrievalResult]]], k: int) -> float:
    if not runs:
        return 0.0

    hits = 0
    for question, results in runs:
        top_sources = {result.chunk.source_uri for result in results[:k]}
        if question.expected_source_uris.intersection(top_sources):
            hits += 1
    return hits / len(runs)


def mean_reciprocal_rank(runs: list[tuple[EvalQuestion, list[RetrievalResult]]]) -> float:
    if not runs:
        return 0.0

    return sum(reciprocal_rank(question, results) for question, results in runs) / len(runs)


def reciprocal_rank(question: EvalQuestion, results: list[RetrievalResult]) -> float:
    for index, result in enumerate(results, start=1):
        if result.chunk.source_uri in question.expected_source_uris:
            return 1 / index
    return 0.0
