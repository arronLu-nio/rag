from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.domain import AnswerTrace, Citation, QAAnswer, RetrievalResult
from app.ports.contracts import ChatModel


class DeepSeekChatModel(ChatModel):
    """DeepSeek 真实生成模型适配器。

    它只负责根据已检索到的授权上下文生成答案；引用仍然由我们自己的 chunk 生成。
    """

    def __init__(self, api_key: str, api_base: str, model: str) -> None:
        self.name = model
        self.client = ChatDeepSeek(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=0,
        )

    async def answer(self, query: str, contexts: list[RetrievalResult]) -> QAAnswer:
        if not contexts:
            return QAAnswer(
                answer="未在当前用户有权限访问的知识库中找到可靠依据，暂不回答。",
                citations=[],
                confidence=0.0,
                trace=AnswerTrace(
                    query=query,
                    rewritten_query=query,
                    user_subjects=set(),
                    retrieval_count=0,
                    reranked_count=0,
                    used_model=self.name,
                    refusal_reason="no_authorized_context",
                ),
            )

        top_contexts = contexts[:3]
        citations = [
            Citation(
                document_id=result.chunk.document_id,
                chunk_id=result.chunk.id,
                title=result.chunk.title,
                source_uri=result.chunk.source_uri,
                page=result.chunk.page,
                quote=result.chunk.text[:180],
            )
            for result in top_contexts
        ]
        context_text = "\n\n".join(
            f"[{index}] 来源：{result.chunk.title}\n{result.chunk.text}"
            for index, result in enumerate(top_contexts, start=1)
        )
        response = await self.client.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是企业知识库问答助手。只能依据用户有权限访问的上下文回答。"
                        "如果上下文不足以回答，就明确说未找到可靠依据。"
                        "回答要简洁，并尽量指出依据来自哪些上下文编号。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"问题：{query}\n\n"
                        f"已授权上下文：\n{context_text}\n\n"
                        "请基于以上上下文回答。"
                    )
                ),
            ]
        )
        return QAAnswer(
            answer=str(response.content),
            citations=citations,
            confidence=min(
                0.95,
                max(result.rerank_score or result.score for result in top_contexts),
            ),
            trace=AnswerTrace(
                query=query,
                rewritten_query=query,
                user_subjects=set(),
                retrieval_count=len(contexts),
                reranked_count=len(top_contexts),
                used_model=self.name,
                retrieved_chunk_ids=[result.chunk.id for result in contexts],
            ),
        )
