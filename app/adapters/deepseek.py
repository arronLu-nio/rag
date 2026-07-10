import logging

from langchain_deepseek import ChatDeepSeek

from app.domain import AnswerTrace, Citation, QAAnswer, RetrievalResult
from app.ports.contracts import ChatModel
from app.prompts.qa import build_qa_messages, select_contexts
from app.settings import settings

logger = logging.getLogger(__name__)


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

        top_contexts = select_contexts(
            contexts,
            max_contexts=settings.max_contexts,
            max_context_chars=settings.max_context_chars,
        )
        if not top_contexts:
            return QAAnswer(
                answer="未在当前用户有权限访问的知识库中找到可靠依据，暂不回答。",
                citations=[],
                confidence=0.0,
                trace=AnswerTrace(
                    query=query,
                    rewritten_query=query,
                    user_subjects=set(),
                    retrieval_count=len(contexts),
                    reranked_count=0,
                    used_model=self.name,
                    refusal_reason="no_context_after_trimming",
                    retrieved_chunk_ids=[result.chunk.id for result in contexts],
                ),
            )

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
        print(f"============={self.name}=============")
        response = await self.client.ainvoke(build_qa_messages(query, top_contexts))
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
