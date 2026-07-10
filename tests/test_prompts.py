from app.domain import ACL, Chunk, RetrievalResult
from app.prompts.qa import build_context_text, select_contexts


def make_result(text: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            document_id="doc-1",
            text=text,
            acl=ACL(tenant_id="t1", space_id="it", allowed_subjects={"user:bob"}),
            source_uri="manual://it",
            title="IT制度",
        ),
        score=score,
        source="test",
    )


def test_select_contexts_limits_count():
    contexts = [make_result("第一段"), make_result("第二段"), make_result("第三段")]

    selected = select_contexts(contexts, max_contexts=2, max_context_chars=100)

    assert [result.chunk.text for result in selected] == ["第一段", "第二段"]


def test_select_contexts_limits_total_chars_and_truncates_last_chunk():
    contexts = [make_result("12345"), make_result("abcdef")]

    selected = select_contexts(contexts, max_contexts=3, max_context_chars=8)

    assert [result.chunk.text for result in selected] == ["12345", "abc"]
    assert build_context_text(selected).count("来源：IT制度") == 2


def test_select_contexts_returns_empty_when_limit_is_zero():
    assert select_contexts([make_result("内容")], max_contexts=0, max_context_chars=100) == []
