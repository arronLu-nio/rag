from app.domain import ACL, Chunk, RetrievalResult
from app.evaluation.retrieval import (
    EvalQuestion,
    mean_reciprocal_rank,
    recall_at_k,
)


def test_recall_and_mrr_handle_empty_runs():
    assert recall_at_k([], 3) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_recall_at_k_counts_expected_source_uri():
    question = EvalQuestion(query="VPN 怎么申请？", expected_source_uris={"source://a"})
    result = RetrievalResult(
        chunk=Chunk(
            document_id="doc",
            text="VPN 账号申请需要主管审批。",
            acl=ACL(tenant_id="eval", space_id="kb", allowed_subjects={"user:evaluator"}),
            source_uri="source://a",
            title="IT制度",
        ),
        score=1.0,
        source="test",
    )

    assert recall_at_k([(question, [result])], 1) == 1.0
    assert mean_reciprocal_rank([(question, [result])]) == 1.0
