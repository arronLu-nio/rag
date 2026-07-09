from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.domain import QAAnswer, RetrievalResult
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
    answer: QAAnswer


def build_qa_graph(retriever: Retriever, reranker: Reranker, chat_model: ChatModel):
    """构建问答图。

    流程：权限解析 -> query rewrite -> 混合召回 -> rerank -> 生成带引用答案。
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
        """按权限召回候选 chunk。"""

        results = await retriever.retrieve(
            query=state["rewritten_query"],
            tenant_id=state["tenant_id"],
            space_id=state["space_id"],
            user_subjects=state["user_subjects"],
            top_k=state.get("top_k", 8),
        )
        return {**state, "retrieved": results}

    async def rerank(state: QAState) -> QAState:
        """对召回结果做二次排序。"""

        reranked = await reranker.rerank(state["rewritten_query"], state["retrieved"])
        return {**state, "reranked": reranked}

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

    graph = StateGraph(QAState)
    graph.add_node("resolve_permissions", resolve_permissions)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_answer", generate_answer)
    graph.set_entry_point("resolve_permissions")
    graph.add_edge("resolve_permissions", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
