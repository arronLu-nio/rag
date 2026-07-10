from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.domain import AnswerTrace, QAAnswer, RetrievalResult
from app.ports.contracts import ChatModel, Reranker, Retriever


class QAState(TypedDict, total=False):
    """LangGraph 问答流程里的共享状态。"""

    query: str
    tenant_id: str
    space_id: str
    user_subjects: set[str]
    top_k: int
    rewritten_query: str
    retrieved: list[RetrievalResult]
    reranked: list[RetrievalResult]
    refusal_reason: str | None
    answer: QAAnswer


def build_qa_graph(
    retriever: Retriever,
    reranker: Reranker,
    chat_model: ChatModel,
    min_retrieval_score: float = 0.2,
    min_rerank_score: float = 0.5,
):
    """构建问答图。

    流程：权限解析 -> query rewrite -> 向量召回 -> 相关性判断 -> rerank -> 生成带引用答案。
    """

    async def resolve_permissions(state: QAState) -> QAState:
        """检查租户/空间信息，并规范化用户身份集合。"""

        if not state.get("tenant_id") or not state.get("space_id"):
            raise ValueError("tenant_id and space_id are required")
        return {**state, "user_subjects": set(state.get("user_subjects", set()))}

    async def rewrite_query(state: QAState) -> QAState:
        """查询改写节点。

        第一版只做 strip；以后可以在这里接多轮上下文改写、同义词扩展或术语归一化。
        """

        query = state["query"].strip()
        return {**state, "rewritten_query": query}

    async def retrieve(state: QAState) -> QAState:
        """按权限召回候选 chunk，并拦截低相关度问题。"""

        results = await retriever.retrieve(
            query=state["rewritten_query"],
            tenant_id=state["tenant_id"],
            space_id=state["space_id"],
            user_subjects=state["user_subjects"],
            top_k=state.get("top_k", 8),
        )
        if not results:
            return {**state, "retrieved": [], "refusal_reason": "no_authorized_context"}
        vector_scores = [
            result.vector_score
            for result in results
            if result.vector_score is not None
        ]
        if not vector_scores and all(result.source != "hybrid-rrf" for result in results):
            # 兼容直接使用 MilvusRetriever 或测试替身的场景。
            vector_scores = [result.score for result in results]
        top_vector_score = max(vector_scores, default=0.0)
        if top_vector_score < min_retrieval_score:
            print(
                "=============召回拒答==============\n"
                f"top_milvus_score={top_vector_score:.4f} "
                f"threshold={min_retrieval_score:.4f}\n"
                "reason=low_retrieval_score\n"
                "================================="
            )
            return {**state, "retrieved": results, "refusal_reason": "low_retrieval_score"}
        return {**state, "retrieved": results, "refusal_reason": None}

    async def rerank(state: QAState) -> QAState:
        """对召回结果做二次排序，并拦截低相关度候选。"""

        reranked = await reranker.rerank(state["rewritten_query"], state["retrieved"])
        if not reranked:
            return {**state, "reranked": [], "refusal_reason": "no_reranked_context"}
        if (reranked[0].rerank_score or 0.0) < min_rerank_score:
            print(
                "=============重排拒答==============\n"
                f"top_rerank_score={reranked[0].rerank_score:.4f} "
                f"threshold={min_rerank_score:.4f}\n"
                "reason=low_rerank_score\n"
                "================================="
            )
            return {
                **state,
                "reranked": reranked,
                "refusal_reason": "low_rerank_score",
            }
        return {**state, "reranked": reranked, "refusal_reason": None}

    async def generate_answer(state: QAState) -> QAState:
        """生成最终答案，并把流程信息写入 trace。"""

        answer = await chat_model.answer(state["rewritten_query"], state["reranked"])
        trace = answer.trace.model_copy(
            update={
                "query": state["query"],
                "rewritten_query": state["rewritten_query"],
                "user_subjects": state["user_subjects"],
                "retrieval_count": len(state["retrieved"]),
                "reranked_count": len(state["reranked"]),
                "used_model": chat_model.name,
                "retrieved_chunk_ids": [result.chunk.id for result in state["retrieved"]],
            }
        )
        return {**state, "answer": answer.model_copy(update={"trace": trace})}

    async def generate_refusal(state: QAState) -> QAState:
        """召回为空或最高分过低时，直接拒答，不调用 rerank 和 LLM。"""

        reason = state["refusal_reason"]
        answer = QAAnswer(
            answer="未在当前用户有权限访问的知识库中找到可靠依据，暂不回答。",
            citations=[],
            confidence=0.0,
            trace=AnswerTrace(
                query=state["query"],
                rewritten_query=state["rewritten_query"],
                user_subjects=state["user_subjects"],
                retrieval_count=len(state["retrieved"]),
                reranked_count=0,
                used_model="retrieval-guard",
                refusal_reason=reason,
                retrieved_chunk_ids=[result.chunk.id for result in state["retrieved"]],
            ),
        )
        return {**state, "answer": answer}

    def route_after_retrieve(state: QAState) -> str:
        """低相关度或无权限上下文直接走拒答节点。"""

        return "generate_refusal" if state.get("refusal_reason") else "rerank"

    def route_after_rerank(state: QAState) -> str:
        """重排结果仍不相关时，跳过 LLM。"""

        return "generate_refusal" if state.get("refusal_reason") else "generate_answer"

    graph = StateGraph(QAState)
    graph.add_node("resolve_permissions", resolve_permissions)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_refusal", generate_refusal)
    graph.set_entry_point("resolve_permissions")
    graph.add_edge("resolve_permissions", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_conditional_edges("rerank", route_after_rerank)
    graph.add_edge("generate_answer", END)
    graph.add_edge("generate_refusal", END)
    return graph.compile()
